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

"""Spawn-safe worker setup and execution evidence for TP experiments."""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

from examples.aics_qwen35_tp2.memory import allocator_sample, read_jsonl, start_sampler
from examples.aics_qwen35_tp2.metrics import (
    summarize_collectives,
    visible_chips,
    worker_device_plan,
)

EXPECTED_PRECISION = {
    "ACL_PRECISION_MODE": "must_keep_origin_dtype",
    "ALLOW_MATMUL_HF32": "disable",
    "ALLOW_CONV_HF32": "disable",
}


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def configure_precision() -> dict:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_option(dict(EXPECTED_PRECISION))
    row = read_precision()
    if not row["matches_requested"]:
        raise RuntimeError(f"Precision readback differs from requested options: {row}")
    return row


def read_precision() -> dict:
    import torch
    import torch_npu

    options = {}
    for key in EXPECTED_PRECISION:
        value = torch_npu._C._npu_getOption(key)
        options[key] = value.decode() if isinstance(value, bytes) else value
    row = {
        "options": options,
        "allow_hf32_matmul": torch.npu.matmul.allow_hf32,
        "allow_hf32_conv": torch.npu.conv.allow_hf32,
    }
    row["matches_requested"] = (
        options == EXPECTED_PRECISION
        and not row["allow_hf32_matmul"]
        and not row["allow_hf32_conv"]
    )
    return row


def initialize_worker(server_args) -> None:
    import torch
    import torch_npu  # noqa: F401

    directory = Path(os.environ["AICS_TP_EVIDENCE_DIR"])
    mapping = server_args.mapping
    chips = visible_chips(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""), expected=mapping.world_size
    )
    plan = worker_device_plan(
        rank=mapping.rank,
        nprocs=mapping.nprocs_per_node,
        base_device=server_args.base_gpu_id,
        chips=chips,
    )
    device = plan["logical_device"]
    # Selecting rank one's device must precede ACL precision configuration.
    torch.npu.set_device(device)
    precision = configure_precision()
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    pid = os.getpid()
    identity = {
        **plan,
        "pid": pid,
        "actual_device": torch.npu.current_device(),
        "world_size": mapping.world_size,
        "attention_tp_size": mapping.attn.tp_size,
        "dense_tp_size": mapping.dense.tp_size,
        "lm_head_tp_size": mapping.lm_head.tp_size,
        "attention_group": list(mapping.attn.tp_group),
        "precision": precision,
        "collective_recording": os.environ["AICS_TP_RECORD_COLLECTIVES"] == "on",
        "initialized_monotonic_s": time.perf_counter(),
    }
    save_json(directory / f"rank-{mapping.rank}-identity.json", identity)
    start_sampler(
        directory=directory,
        device=device,
        rank=mapping.rank,
        pid=pid,
        interval_s=float(os.environ["AICS_TP_MEMORY_INTERVAL_S"]),
    )
    events_path = directory / f"rank-{mapping.rank}-events.jsonl"
    original = ModelExecutor.execute_forward_op
    parameter_recorded = False

    def append_event(row: dict) -> None:
        with events_path.open("a") as output:
            output.write(json.dumps({"rank": mapping.rank, "pid": pid, **row}) + "\n")

    @functools.wraps(original)
    def execute(executor, forward_op, *args, **kwargs):
        nonlocal parameter_recorded
        if not parameter_recorded:
            from tokenspeed.runtime.distributed.comm_backend.registry import (
                get_global_backend,
            )

            parameters = [
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "numel": value.numel(),
                    "bytes": value.numel() * value.element_size(),
                }
                for name, value in executor.model_runner.model.named_parameters()
            ]
            save_json(
                directory / f"rank-{mapping.rank}-parameters.json",
                {
                    "rank": mapping.rank,
                    "pid": pid,
                    "parameters": parameters,
                    "communication_backend": type(get_global_backend()).__name__,
                },
            )
            parameter_recorded = True
        start = time.perf_counter()
        row = {
            "event": "forward",
            "start_monotonic_s": start,
            "num_tokens": sum(forward_op.input_lengths),
            "batch_size": len(forward_op.request_ids),
            "num_extends": forward_op.num_extends(),
            "actual_device": torch.npu.current_device(),
        }
        try:
            result = original(executor, forward_op, *args, **kwargs)
            row["success"] = True
            return result
        except Exception as exc:
            row.update(success=False, error=repr(exc))
            raise
        finally:
            row["end_monotonic_s"] = time.perf_counter()
            # Host dispatch span deliberately does not synchronize the NPU.
            append_event(row)
            save_json(
                directory / f"rank-{mapping.rank}-precision-after.json",
                read_precision(),
            )
            append_event(
                {
                    "event": "memory_after_forward",
                    **allocator_sample(device=device, rank=mapping.rank, pid=pid),
                }
            )

    ModelExecutor.execute_forward_op = execute
    if identity["collective_recording"]:
        install_collective_recording(append_event)


def install_collective_recording(append_event) -> None:
    import torch

    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend

    def wrap(name: str, input_index: int) -> None:
        original = getattr(HcclBackend, name)

        @functools.wraps(original)
        def operation(backend, *args, **kwargs):
            value = args[input_index]
            if not isinstance(value, torch.Tensor):
                return original(backend, *args, **kwargs)
            start = time.perf_counter()
            row = {
                "event": "collective",
                "operation": name,
                "input_shape": list(value.shape),
                "dtype": str(value.dtype),
                "payload_bytes": value.numel() * value.element_size(),
                "start_monotonic_s": start,
            }
            try:
                result = original(backend, *args, **kwargs)
                row["success"] = True
                return result
            except Exception as exc:
                row.update(success=False, error=repr(exc))
                raise
            finally:
                row["end_monotonic_s"] = time.perf_counter()
                append_event(row)

        setattr(HcclBackend, name, operation)

    # Count primitive calls once; token_* wrappers eventually invoke these in
    # the uniform TP layout used here. Payload is logical input, not link bytes.
    for name, index in (
        ("all_reduce", 0),
        ("all_gather_single", 1),
        ("reduce_scatter", 0),
        ("all_to_all_single", 1),
    ):
        wrap(name, index)


def install_worker_hooks(*additional_hooks) -> None:
    from multiprocessing.process import BaseProcess

    original = BaseProcess.run

    @functools.wraps(original)
    def run(process):
        target = process._target
        if (
            target is not None
            and target.__module__ == "tokenspeed.runtime.engine.event_loop"
            and target.__name__ == "run_event_loop"
            and "AICS_TP_EVIDENCE_DIR" in os.environ
        ):
            initialize_worker(process._args[0])
            for hook in additional_hooks:
                hook(process._args[0])
        return original(process)

    BaseProcess.run = run


def collect_workers(directory: Path) -> list[dict]:
    workers = []
    for path in sorted(directory.glob("rank-*-identity.json")):
        row = json.loads(path.read_text())
        events = read_jsonl(directory / f"rank-{row['rank']}-events.jsonl")
        forwards = [event for event in events if event["event"] == "forward"]
        good = [
            event for event in forwards if event["success"] and event["num_tokens"] > 0
        ]
        samples = [
            sample
            for sample in read_jsonl(directory / f"rank-{row['rank']}-memory.jsonl")
            if "error" not in sample
        ]
        samples.extend(
            event for event in events if event["event"] == "memory_after_forward"
        )
        precision_after = directory / f"rank-{row['rank']}-precision-after.json"
        parameters_path = directory / f"rank-{row['rank']}-parameters.json"
        parameters = (
            json.loads(parameters_path.read_text()) if parameters_path.exists() else {}
        )
        row.update(
            successful_forward_count=len(good),
            successful_prefill_count=sum(event["num_extends"] > 0 for event in good),
            successful_decode_count=sum(event["num_extends"] == 0 for event in good),
            forward_batch_sizes=sorted({event["batch_size"] for event in good}),
            forward_errors=[event for event in forwards if not event["success"]],
            forward_device_ids=sorted({event["actual_device"] for event in forwards}),
            memory_sample_count=len(samples),
            memory_device_ids=sorted({sample["logical_device"] for sample in samples}),
            precision_after_forward=(
                json.loads(precision_after.read_text())
                if precision_after.exists()
                else None
            ),
            communication_backend=parameters.get("communication_backend"),
            sampled_max_allocated_bytes=max(
                (sample["allocated_bytes"] for sample in samples), default=0
            ),
            collective_summary=summarize_collectives(events),
            parameter_file=str(directory / f"rank-{row['rank']}-parameters.json"),
        )
        workers.append(row)
    return workers
