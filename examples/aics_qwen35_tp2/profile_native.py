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

"""Capture separate CPU+NPU timelines from both tensor-parallel workers."""

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
from dataclasses import dataclass
from pathlib import Path

from examples.aics_qwen35_tp2.hooks import (
    collect_workers,
    install_worker_hooks,
    read_precision,
)
from examples.aics_qwen35_tp2.metrics import validate_worker_evidence, visible_chips
from examples.aics_qwen35_tp2.run import (
    REPO,
    model_identity,
    native_engine_kwargs,
    new_artifact,
    run_group,
    selected_cases,
    sha256,
    write_json,
)


@dataclass
class CaptureState:
    active: bool = False
    owner_thread: int | None = None


CAPTURE = CaptureState()


def run_profile_transition(*, operation, agree, rollback) -> dict:
    """Converge a rank-local profiler operation and roll back peer failures.

    A local error must still participate in the replica decision. Returning
    before the CPU collective leaves healthy peers waiting for a missing rank.
    """
    error = None
    try:
        operation()
    except Exception:
        error = traceback.format_exc()
    try:
        success = agree(error is None)
    except Exception:
        rollback()
        raise
    if success:
        return {
            "success": True,
            "message": "All TP ranks completed the profile operation",
        }
    message = (
        error
        or "Another TP rank failed the profile operation; inspect per-rank status files"
    )
    try:
        rollback()
    except Exception:
        message += "\nCleanup failed:\n" + traceback.format_exc()
    return {"success": False, "message": message}


def install_profile_hooks(server_args) -> None:
    import torch
    import torch_npu

    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.engine.io_struct import ProfileReqOutput
    from tokenspeed.runtime.engine.request_handler import RequestHandler
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    rank = server_args.mapping.rank
    original_forward = ModelExecutor.execute_forward_op

    def reset_state(handler):
        handler.torch_profiler = None
        handler.profile_in_progress = False
        handler.profiler_start_forward_ct = None
        CAPTURE.active = False
        CAPTURE.owner_thread = None

    def start_profile(handler, stage=None):
        directory = Path(handler.profiler_output_dir)
        started = False

        def start_local():
            nonlocal started
            if stage is not None or handler.profile_by_stage:
                raise ValueError("Use one explicit profile window")
            if handler.profiler_activities != ["CPU", "NPU"]:
                raise ValueError("Expected CPU,NPU activities")
            directory.mkdir(parents=True, exist_ok=True)
            callback = torch_npu.profiler.tensorboard_trace_handler(
                str(directory),
                worker_name=f"aics-tp{rank}-{os.getpid()}",
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
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    export_type=torch_npu.profiler.ExportType.Text,
                    data_simplification=False,
                ),
            )
            handler.torch_profiler.start()
            started = True
            CAPTURE.owner_thread = threading.get_ident()
            CAPTURE.active = True
            handler.profile_in_progress = True

        def rollback():
            try:
                if started:
                    handler.torch_profiler.stop()
            finally:
                reset_state(handler)

        # The frontend receives TP0's reply only. Every rank must attempt
        # startup and converge, including a rank whose local API raised.
        result = run_profile_transition(
            operation=start_local,
            agree=handler.converge_replica_decision,
            rollback=rollback,
        )
        write_json(
            directory / f"rank-{rank}.profile-start.json",
            {"rank": rank, "pid": os.getpid(), "profiler_level": "Level1", **result},
        )
        return ProfileReqOutput(**result)

    def stop_profile(handler, stage=None):
        CAPTURE.active = False
        directory = Path(handler.profiler_output_dir)
        trace_path = directory / f"{handler.profile_id}-TP{rank}.trace.json"
        status = {
            "rank": rank,
            "pid": os.getpid(),
            "trace_path": str(trace_path),
            "profiler_level": "Level1",
        }
        stopped = False

        def export_local():
            nonlocal stopped
            if not handler.profile_in_progress or handler.torch_profiler is None:
                raise RuntimeError("No profile active")
            status["npu_precision"] = read_precision()
            status["raw_data_directory"] = handler.torch_profiler.prof_if.prof_path
            handler.torch_profiler.stop()
            stopped = True
            handler.torch_profiler.export_chrome_trace(str(trace_path))
            if not trace_path.is_file() or trace_path.stat().st_size == 0:
                raise RuntimeError("NPU profiler did not export a trace")
            status.update(success=True, message="Local trace export completed")
            write_json(directory / f"rank-{rank}.profile-status.json", status)

        def rollback():
            if not stopped and handler.torch_profiler is not None:
                handler.torch_profiler.stop()

        try:
            result = run_profile_transition(
                operation=export_local,
                agree=handler.converge_replica_decision,
                rollback=rollback,
            )
            status.update(result)
            # This MIN-reduce completes after every rank finishes its export;
            # successful status files also precede that collective, so TP0's
            # reply cannot race TP1's status-file write.
            if not result["success"]:
                write_json(directory / f"rank-{rank}.profile-status.json", status)
            return ProfileReqOutput(**result)
        finally:
            reset_state(handler)

    @functools.wraps(original_forward)
    def forward(executor, forward_op, *args, **kwargs):
        if not CAPTURE.active:
            return original_forward(executor, forward_op, *args, **kwargs)
        child = threading.get_ident() != CAPTURE.owner_thread
        if child:
            torch_npu.profiler.profile.enable_profiler_in_child_thread()
        stage = "prefill" if forward_op.num_extends() else "decode"
        try:
            with torch.profiler.record_function(f"aics.{stage}"):
                return original_forward(executor, forward_op, *args, **kwargs)
        finally:
            if child:
                # This torch_npu API synchronizes the NPU. Both ranks use the
                # same capture mode; ordinary benchmark latency stays separate.
                torch_npu.profiler.profile.disable_profiler_in_child_thread()

    def tag_collective(name):
        original = getattr(HcclBackend, name)

        @functools.wraps(original)
        def collective(backend, *args, **kwargs):
            if not CAPTURE.active:
                return original(backend, *args, **kwargs)
            with torch.profiler.record_function(f"aics.comm.{name}"):
                return original(backend, *args, **kwargs)

        setattr(HcclBackend, name, collective)

    RequestHandler.start_profile = start_profile
    RequestHandler.stop_profile = stop_profile
    ModelExecutor.execute_forward_op = forward
    for name in (
        "all_reduce",
        "all_gather_single",
        "reduce_scatter",
        "all_to_all_single",
    ):
        tag_collective(name)


if __name__ == "__mp_main__":
    import torch_npu  # noqa: F401

    install_worker_hooks(install_profile_hooks)


def execute_profile(args: argparse.Namespace, artifact: dict) -> None:
    visible_chips(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""), expected=args.tp_size
    )
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.entrypoints.engine import Engine

    if (
        Path(sys.modules[Engine.__module__].__file__).resolve()
        != REPO / "python/tokenspeed/runtime/entrypoints/engine.py"
    ):
        raise ValueError("Engine did not import from this checkout")
    cases = selected_cases(args)
    if len(cases) != 1:
        raise ValueError("Profile one input case with --case-ids")
    case = {**cases[0], "max_new_tokens": args.profile_output_tokens}
    if case["input_tokens"] + case["max_new_tokens"] > args.max_model_len:
        raise ValueError("Profile request exceeds context length")
    output = Path(args.output).resolve()
    directory = output.parent / f"{output.stem}.traces-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    evidence_dir = directory / "workers"
    evidence_dir.mkdir()
    env = {
        "AICS_TP_EVIDENCE_DIR": str(evidence_dir),
        "AICS_TP_MEMORY_INTERVAL_S": str(args.memory_interval_s),
        "AICS_TP_RECORD_COLLECTIVES": args.record_collectives,
    }
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    engine_args = native_engine_kwargs(args, output_logprobs=False)
    artifact.update(
        engine_args=engine_args,
        cases_sha256=sha256(args.cases),
        model_identity=model_identity(args.model),
        trace_directory=str(directory),
        profiling_scope="Unprofiled warmup followed by one profiled concurrent request group; per-forward child-profiler disable synchronizes NPU. Capture durations are diagnostic, not benchmark latency.",
    )
    write_json(args.output, artifact)
    engine, active = None, False
    try:
        startup = time.perf_counter()
        engine = Engine(**engine_args)
        artifact["engine_startup_s"] = time.perf_counter() - startup
        warmup, _ = run_group(
            engine,
            case,
            concurrency=args.concurrency,
            phase="warmup",
            group_id="profile-warmup",
        )
        artifact["results"].extend(warmup)
        engine.llm.run(
            engine.tokenizer_manager.start_profile(
                output_dir=str(directory),
                activities=["CPU", "NPU"],
                with_stack=False,
                record_shapes=False,
                profile_by_stage=False,
                profile_id="aics-tp-profile",
            )
        )
        active = True
        rows, group = run_group(
            engine,
            case,
            concurrency=args.concurrency,
            phase="profiled",
            group_id="profile-capture",
        )
        artifact["results"].extend(rows)
        artifact["profiled_group"] = group
        engine.stop_profile()
        active = False
        statuses = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("rank-*.profile-status.json"))
        ]
        artifact["profiler_worker_status"] = statuses
        if sorted(row["rank"] for row in statuses) != list(
            range(args.tp_size)
        ) or not all(
            row.get("success") and row["npu_precision"]["matches_requested"]
            for row in statuses
        ):
            raise RuntimeError(
                "All requested ranks must export successful NPU traces with matching precision"
            )
        traces = {
            row["rank"]: json.loads(Path(row["trace_path"]).read_text())
            for row in statuses
        }
        from examples.aics_qwen35_tp2.profile_summary import summarize_rank_traces

        artifact["trace_summary"] = summarize_rank_traces(traces)
        if not all(
            row["stage_evidence_present"] for row in artifact["trace_summary"]["ranks"]
        ):
            raise RuntimeError(
                "Every rank must have flow-attributed NPU computation in both prefill and decode"
            )
        artifact["communication_evidence_present"] = all(
            row["communication_evidence_present"]
            for row in artifact["trace_summary"]["ranks"]
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
        artifact["workers"] = collect_workers(evidence_dir)
        artifact["worker_evidence"] = validate_worker_evidence(
            artifact["workers"], tp_size=args.tp_size
        )
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not artifact["worker_evidence"]["passed"]:
        raise RuntimeError(f"Worker evidence failed: {artifact['worker_evidence']}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--suite", choices=("validation", "benchmark"), required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--sampling-backend", required=True)
    parser.add_argument("--overlap-schedule", choices=("on", "off"), required=True)
    parser.add_argument("--record-collectives", choices=("on", "off"), required=True)
    parser.add_argument("--prefill-chunk-tokens", type=int, required=True)
    parser.add_argument("--memory-utilization", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    parser.add_argument("--memory-interval-s", type=float, default=0.2)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=180)
    parser.add_argument("--profile-output-tokens", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.set_defaults(command="profile")
    return parser


def main() -> int:
    args = parser().parse_args()
    artifact = new_artifact(args)
    write_json(args.output, artifact)
    try:
        if args.concurrency < 1 or not 2 <= args.profile_output_tokens <= 16:
            raise ValueError(
                "Concurrency must be positive and profile output length 2..16"
            )
        if (
            not 0 < args.memory_utilization < 1
            or min(
                args.memory_interval_s,
                args.prefill_chunk_tokens,
                args.max_model_len,
                args.max_total_tokens,
                args.distributed_timeout_seconds,
            )
            <= 0
        ):
            raise ValueError("Invalid memory or capacity settings")
        execute_profile(args, artifact)
        artifact["status"] = "completed"
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["errors"].append(
            {"exception": repr(exc), "traceback": traceback.format_exc()}
        )
        traceback.print_exc()
    finally:
        write_json(args.output, artifact)
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
