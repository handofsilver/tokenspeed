# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Profile one warmed native request; keep this separate from latency benchmarks."""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from metrics import StreamRecord
from precision import (
    collect_worker_precision,
    configure_npu_precision,
    install_spawned_worker_hooks,
    read_npu_precision,
)
from run import (
    MODEL,
    REPO,
    native_engine_kwargs,
    new_artifact,
    require_one_chip,
    selected_cases,
    sha256,
    write_json,
)


@dataclass
class ProfileState:
    active: bool = False
    owner_thread: int | None = None


STATE = ProfileState()


def install_npu_profile_hooks() -> None:
    """Opt-in child-process adapters around the existing profile control RPC."""
    if os.environ.get("AICS_NPU_PROFILE") != "1":
        return
    import torch
    import torch_npu

    from tokenspeed.runtime.engine.io_struct import ProfileReqOutput
    from tokenspeed.runtime.engine.request_handler import RequestHandler
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    original_forward = ModelExecutor.execute_forward_op

    def start_profile(handler, stage=None):
        if stage is not None or handler.profile_by_stage:
            return ProfileReqOutput(
                success=False, message="Use one explicitly started profile window"
            )
        if handler.profiler_activities != ["CPU", "NPU"]:
            return ProfileReqOutput(
                success=False, message="AICS profiler requires exactly CPU,NPU"
            )
        try:
            directory = Path(handler.profiler_output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            # Keep raw data even if export/summary fails. The callback configures
            # a persistent directory but does not run a second analysis pass.
            callback = torch_npu.profiler.tensorboard_trace_handler(
                str(directory),
                worker_name=f"aics-{os.getpid()}",
                analyse_flag=False,
            )
            handler.torch_profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=callback,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
            )
            handler.torch_profiler.start()
            STATE.owner_thread = threading.get_ident()
            STATE.active = True
            handler.profile_in_progress = True
            return ProfileReqOutput(
                success=True, message="AICS CPU+NPU profile started"
            )
        except Exception:
            return ProfileReqOutput(success=False, message=traceback.format_exc())

    def stop_profile(handler, stage=None):
        if not handler.profile_in_progress or handler.torch_profiler is None:
            return ProfileReqOutput(success=False, message="No AICS profile is active")
        STATE.active = False
        directory = Path(handler.profiler_output_dir)
        status = {
            "pid": os.getpid(),
            "raw_data_directory": handler.torch_profiler.prof_if.prof_path,
            "npu_precision": read_npu_precision(),
        }
        trace_path = (
            directory / f"{handler.profile_id}-{handler.profile_rank_tag}.trace.json"
        )
        try:
            handler.torch_profiler.stop()
            handler.torch_profiler.export_chrome_trace(str(trace_path))
            status["trace_path"] = str(trace_path)
            if not trace_path.is_file() or trace_path.stat().st_size == 0:
                raise RuntimeError(
                    "NPU profiler did not produce a Chrome trace; raw data retained"
                )
            status["success"] = True
            return ProfileReqOutput(success=True, message=str(trace_path))
        except Exception:
            status.update(success=False, error=traceback.format_exc())
            return ProfileReqOutput(success=False, message=status["error"])
        finally:
            write_json(directory / f"worker-{os.getpid()}.profile-status.json", status)
            handler.torch_profiler = None
            handler.profile_in_progress = False
            handler.profiler_start_forward_ct = None
            STATE.owner_thread = None

    @functools.wraps(original_forward)
    def tagged_forward(executor, forward_op, *args, **kwargs):
        if not STATE.active:
            return original_forward(executor, forward_op, *args, **kwargs)
        child_thread = threading.get_ident() != STATE.owner_thread
        if child_thread:
            # ModelExecutor runs on the device thread. The installed torch_npu
            # API requires explicit enablement there for CPU-to-device flows.
            torch_npu.profiler.profile.enable_profiler_in_child_thread()
        stage = "prefill" if forward_op.num_extends() else "decode"
        try:
            with torch.profiler.record_function(f"aics.{stage}"):
                return original_forward(executor, forward_op, *args, **kwargs)
        finally:
            if child_thread:
                # This API synchronizes the NPU. These traces are diagnostic;
                # their request latency must never replace benchmark timings.
                torch_npu.profiler.profile.disable_profiler_in_child_thread()

    RequestHandler.start_profile = start_profile
    RequestHandler.stop_profile = stop_profile
    ModelExecutor.execute_forward_op = tagged_forward


if __name__ == "__mp_main__":
    # Registration must precede target imports; compiler settings must follow
    # bootstrap, when CANN may safely spawn its helper processes.
    import torch_npu  # noqa: F401

    install_spawned_worker_hooks(install_npu_profile_hooks)


def event_key(event: dict) -> tuple:
    return str(event["pid"]), str(event["tid"]), round(float(event["ts"]), 3)


def containing_stage(event: dict, stages: list[dict]) -> str:
    timestamp = float(event["ts"])
    for stage in stages:
        if (
            str(event["pid"]) == str(stage["pid"])
            and str(event["tid"]) == str(stage["tid"])
            and float(stage["ts"])
            <= timestamp
            < float(stage["ts"]) + float(stage["dur"])
        ):
            return stage["name"].removeprefix("aics.")
    return "unattributed"


def device_task_kind(event: dict) -> str | None:
    """Classify CANN device work from task metadata, not flow connectivity."""
    task_type = str(event.get("args", {}).get("Task Type", "")).upper()
    if task_type.startswith(("SDMA", "RDMA", "DMA", "MEMCPY")):
        return "transfer"
    if task_type.startswith("KERNEL_") or task_type in {
        "AI_CORE",
        "AI_CPU",
        "AI_VECTOR",
        "MIX_AIC",
        "MIX_AIV",
    }:
        return "kernel"
    return None


def summarize_trace(trace: list | dict, *, top_n: int) -> dict:
    """Reduce Chrome events without treating asynchronous kernels as CPU ranges."""
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    complete = [
        event
        for event in events
        if event.get("ph") == "X" and float(event.get("dur", 0)) >= 0
    ]
    stages = [
        event
        for event in complete
        if event.get("name") in ("aics.prefill", "aics.decode")
    ]
    cpu = [event for event in complete if event.get("cat") == "cpu_op"]
    # For properly nested CPU ranges, subtract direct child ranges exactly once.
    # This is self thread-wall time (includes waiting), not hardware CPU usage.
    cpu_self = {id(event): float(event["dur"]) for event in cpu}
    threads = defaultdict(list)
    for event in cpu:
        threads[(str(event["pid"]), str(event["tid"]))].append(event)
    partial_overlaps = 0
    for group in threads.values():
        stack = []
        for event in sorted(
            group, key=lambda row: (float(row["ts"]), -float(row["dur"]))
        ):
            start, end = float(event["ts"]), float(event["ts"]) + float(event["dur"])
            while stack and start >= float(stack[-1]["ts"]) + float(stack[-1]["dur"]):
                stack.pop()
            if stack:
                parent = stack[-1]
                if end <= float(parent["ts"]) + float(parent["dur"]) + 0.001:
                    cpu_self[id(parent)] -= float(event["dur"])
                else:
                    partial_overlaps += 1
                    stack.clear()
            stack.append(event)
    cpu_groups = {}
    for event in cpu:
        if event["name"].startswith("aics."):
            continue
        stage = containing_stage(event, stages)
        key = (stage, event["name"])
        row = cpu_groups.setdefault(
            key,
            {
                "stage": stage,
                "name": event["name"],
                "count": 0,
                "inclusive_thread_us": 0.0,
                "self_thread_us": 0.0,
            },
        )
        row["count"] += 1
        row["inclusive_thread_us"] += float(event["dur"])
        row["self_thread_us"] += max(0.0, cpu_self[id(event)])

    starts = {
        str(event["id"]): event
        for event in events
        if event.get("name") == "torch_to_npu" and event.get("ph") == "s"
    }
    finishes = [
        event
        for event in events
        if event.get("name") == "torch_to_npu" and event.get("ph") == "f"
    ]
    linked = {}
    for finish in finishes:
        start = starts.get(str(finish["id"]))
        if start is not None:
            linked[event_key(finish)] = containing_stage(start, stages)
    kernels = [event for event in complete if device_task_kind(event) == "kernel"]
    transfers = [event for event in complete if device_task_kind(event) == "transfer"]

    def group_device_events(device_events):
        groups = {}
        for event in device_events:
            stage = linked.get(event_key(event), "unattributed")
            task_type = event["args"]["Task Type"]
            key = stage, event["name"], task_type
            row = groups.setdefault(
                key,
                {
                    "stage": stage,
                    "name": event["name"],
                    "task_type": task_type,
                    "count": 0,
                    "duration_sum_us": 0.0,
                },
            )
            row["count"] += 1
            row["duration_sum_us"] += float(event["dur"])
        return groups

    kernel_groups = group_device_events(kernels)
    transfer_groups = group_device_events(transfers)
    stage_summary = {}
    for name in ("prefill", "decode", "unattributed"):
        matching_ranges = [event for event in stages if event["name"] == f"aics.{name}"]
        matching_kernels = [
            row for row in kernel_groups.values() if row["stage"] == name
        ]
        matching_transfers = [
            row for row in transfer_groups.values() if row["stage"] == name
        ]
        stage_summary[name] = {
            "forward_ranges": len(matching_ranges),
            "host_forward_range_sum_us": sum(
                float(event["dur"]) for event in matching_ranges
            ),
            "kernel_count": sum(row["count"] for row in matching_kernels),
            "kernel_duration_sum_us": sum(
                row["duration_sum_us"] for row in matching_kernels
            ),
            "transfer_count": sum(row["count"] for row in matching_transfers),
            "transfer_duration_sum_us": sum(
                row["duration_sum_us"] for row in matching_transfers
            ),
            "top_cpu_ops_by_self_thread_time": sorted(
                (row for row in cpu_groups.values() if row["stage"] == name),
                key=lambda row: row["self_thread_us"],
                reverse=True,
            )[:top_n],
            "top_kernels_by_duration_sum": sorted(
                matching_kernels,
                key=lambda row: row["duration_sum_us"],
                reverse=True,
            )[:top_n],
            "top_transfers_by_duration_sum": sorted(
                matching_transfers,
                key=lambda row: row["duration_sum_us"],
                reverse=True,
            )[:top_n],
        }
    total_kernel_duration = sum(float(event["dur"]) for event in kernels)
    unattributed = stage_summary["unattributed"]
    cpu_stage_ranges_present = all(
        stage_summary[name]["forward_ranges"] > 0 for name in ("prefill", "decode")
    )
    return {
        "trace_event_count": len(events),
        "complete_event_categories": dict(
            Counter(event.get("cat", "") for event in complete)
        ),
        "cpu_event_count": len(cpu),
        "aten_operator_count": sum(event["name"].startswith("aten::") for event in cpu),
        "matrix_operator_count": sum(
            event["name"]
            in {"aten::mm", "aten::bmm", "aten::matmul", "aten::linear", "aten::addmm"}
            for event in cpu
        ),
        "npu_kernel_event_count": len(kernels),
        "npu_transfer_event_count": len(transfers),
        "npu_transfer_duration_sum_us": sum(float(event["dur"]) for event in transfers),
        "device_task_type_counts": dict(
            Counter(
                event["args"]["Task Type"]
                for event in complete
                if "Task Type" in event.get("args", {})
            )
        ),
        "unclassified_device_task_type_counts": dict(
            Counter(
                event["args"]["Task Type"]
                for event in complete
                if "Task Type" in event.get("args", {})
                and device_task_kind(event) is None
            )
        ),
        "cpu_to_npu_flow_count": len(finishes),
        "linked_kernel_count": sum(event_key(event) in linked for event in kernels),
        "linked_transfer_count": sum(event_key(event) in linked for event in transfers),
        "kernel_attribution": {
            "total_count": len(kernels),
            "unattributed_count": unattributed["kernel_count"],
            "unattributed_count_fraction": (
                unattributed["kernel_count"] / len(kernels) if kernels else None
            ),
            "total_duration_sum_us": total_kernel_duration,
            "unattributed_duration_sum_us": unattributed["kernel_duration_sum_us"],
            "unattributed_duration_fraction": (
                unattributed["kernel_duration_sum_us"] / total_kernel_duration
                if total_kernel_duration > 0
                else None
            ),
        },
        "cpu_partial_overlap_count": partial_overlaps,
        "cpu_stage_ranges_present": cpu_stage_ranges_present,
        "stage_evidence_present": cpu_stage_ranges_present
        and all(
            stage_summary[name]["kernel_count"] > 0 for name in ("prefill", "decode")
        ),
        "stages": stage_summary,
        "limitations": [
            "CPU self values are estimated thread-wall durations, including waits; partial overlaps weaken exclusivity.",
            "Kernel duration sums may overlap and are not wall latency or device utilization.",
            "Device kernels and transfers use Task Type metadata; a flow-linked copy is not kernel evidence.",
            "Kernel and transfer stages use torch_to_npu flow source ranges, never CPU/device timestamp overlap.",
            "Device events lacking recognized task metadata remain unclassified; retain task-type/category counts and raw trace.",
        ],
    }


def generate_short(engine, case: dict, *, outputs: int, phase: str) -> dict:
    started = time.perf_counter()
    collector = StreamRecord(start_s=started)
    for frame in engine.generate(
        input_ids=case["input_ids"],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": outputs,
            "ignore_eos": True,
        },
        return_logprob=False,
        stream=True,
    ):
        collector.add(frame, arrival_s=time.perf_counter())
    row = {
        **case,
        **collector.finish(end_s=time.perf_counter()),
        "phase": phase,
        "requested_profile_output_tokens": outputs,
    }
    if row["actual_output_tokens"] != outputs:
        raise ValueError(
            "Profile request stopped before the requested number of tokens"
        )
    return row


def execute_profile(args, artifact):
    require_one_chip()
    import torch_npu  # noqa: F401

    artifact["npu_precision"] = configure_npu_precision()

    from tokenspeed.runtime.entrypoints.engine import Engine

    source = Path(sys.modules[Engine.__module__].__file__).resolve()
    if source != REPO / "python/tokenspeed/runtime/entrypoints/engine.py":
        raise ValueError(f"Expected Engine from this checkout; imported {source}")
    cases = selected_cases(args)
    if len(cases) != 1:
        raise ValueError("Profile exactly one case; pass one --case-ids value")
    case = cases[0]
    if case["input_tokens"] + args.max_new_tokens > args.max_model_len:
        raise ValueError("Profile request exceeds model context")
    trace_dir = (
        Path(args.output).resolve().parent
        / f"{Path(args.output).stem}.traces-{uuid.uuid4().hex[:8]}"
    )
    trace_dir.mkdir(parents=True)
    os.environ["AICS_NPU_PROFILE"] = "1"
    os.environ["AICS_NPU_PRECISION_DIR"] = str(trace_dir)
    kwargs = native_engine_kwargs(args, output_logprobs=False)
    artifact.update(
        engine_args=kwargs,
        cases_sha256=sha256(args.cases),
        trace_directory=str(trace_dir),
    )
    artifact["profiling_scope"] = (
        "One unprofiled warmup then one profiled request. Examples-only worker hook uses "
        "the existing profile control RPC with torch_npu CPU+NPU capture; per-forward child-thread "
        "profiler disable synchronizes the NPU. Profile request times are diagnostic, not benchmark results."
    )
    engine = None
    active = False
    try:
        engine = Engine(**kwargs)
        artifact["results"].append(
            generate_short(engine, case, outputs=args.max_new_tokens, phase="warmup")
        )
        write_json(args.output, artifact)
        engine.llm.run(
            engine.tokenizer_manager.start_profile(
                output_dir=str(trace_dir),
                activities=["CPU", "NPU"],
                with_stack=False,
                record_shapes=False,
                profile_by_stage=False,
                profile_id="aics-one-request",
            )
        )
        active = True
        artifact["results"].append(
            generate_short(engine, case, outputs=args.max_new_tokens, phase="profiled")
        )
        engine.stop_profile()
        active = False
        for path in trace_dir.glob("*.profile-status.json"):
            status = json.loads(path.read_text())
            if not status.get("npu_precision", {}).get("matches_requested"):
                raise RuntimeError(f"Profiler worker precision mismatch: {path}")
        summaries = []
        for path in sorted(trace_dir.glob("*.trace.json")):
            try:
                summaries.append(
                    {
                        "trace_path": str(path),
                        **summarize_trace(
                            json.loads(path.read_text()), top_n=args.top_n
                        ),
                    }
                )
            except Exception:
                summaries.append(
                    {"trace_path": str(path), "summary_error": traceback.format_exc()}
                )
        artifact["trace_summaries"] = summaries
        if not any(
            summary.get("npu_kernel_event_count", 0) > 0
            and summary.get("matrix_operator_count", 0) > 0
            and summary.get("stage_evidence_present", False)
            for summary in summaries
        ):
            raise RuntimeError(
                "Prefill and decode must each contain a CPU forward range and flow-attributed NPU kernels; "
                "inspect retained traces and profiler status files"
            )
    finally:
        if engine is not None:
            if active:
                try:
                    engine.stop_profile()
                except Exception:
                    artifact["errors"].append(
                        {"stop_profile_error": traceback.format_exc()}
                    )
            engine.shutdown()
        os.environ.pop("AICS_NPU_PROFILE", None)
        os.environ.pop("AICS_NPU_PRECISION_DIR", None)
        artifact["worker_precision"] = collect_worker_precision(trace_dir)
        artifact["profiler_status_files"] = [
            str(path) for path in sorted(trace_dir.glob("*.profile-status.json"))
        ]
        artifact["profiler_worker_status"] = [
            json.loads(path.read_text())
            for path in sorted(trace_dir.glob("*.profile-status.json"))
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--case-ids", default="A17", help="Exactly one prepared case")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--sampling-backend", required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    parser.add_argument("--memory-utilization", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.set_defaults(command="profile", suite="validation")
    args = parser.parse_args()
    artifact = new_artifact(args)
    write_json(args.output, artifact)
    try:
        if not 2 <= args.max_new_tokens <= 16:
            raise ValueError(
                "Use 2..16 output tokens to keep this diagnostic trace bounded"
            )
        if args.top_n <= 0:
            raise ValueError("top-n must be positive")
        execute_profile(args, artifact)
        artifact["status"] = "completed"
    except Exception:
        artifact["status"] = "failed"
        artifact["errors"].append({"traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        write_json(args.output, artifact)
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
