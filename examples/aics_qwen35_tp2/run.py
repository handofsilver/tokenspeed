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

"""Prepare inputs and run native TP1/TP2 correctness and throughput experiments."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from examples.aics_qwen35_tp2.hooks import collect_workers, install_worker_hooks
from examples.aics_qwen35_tp2.memory import summarize_memory
from examples.aics_qwen35_tp2.metrics import (
    StreamRecord,
    benchmark_summary,
    concurrent_group,
    validate_worker_evidence,
    visible_chips,
)

REPO = Path(__file__).resolve().parents[2]

if __name__ == "__mp_main__":
    # Register dispatch while unpickling; device initialization belongs after
    # bootstrap, when CANN may safely start its helper processes.
    import torch_npu  # noqa: F401

    install_worker_hooks()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_identity(model: str) -> dict:
    path = Path(model)
    index = path / "model.safetensors.index.json"
    # Revision and full weight checksums are kept in the download manifest;
    # these file identities also catch accidentally mixed config/tokenizer files.
    return {
        "model_config_sha256": sha256(path / "config.json"),
        "tokenizer_sha256": sha256(path / "tokenizer.json"),
        "weight_index_sha256": sha256(index) if index.exists() else None,
        "weight_files": [
            {"name": item.name, "bytes": item.stat().st_size}
            for item in sorted(path.glob("*.safetensors"))
        ],
    }


def environment() -> dict:
    versions = {}
    for package in ("torch", "torch-npu", "transformers", "tokenspeed-scheduler"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "ls-files",
            "-z",
            "--modified",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        check=False,
    )
    files = {}
    if changed.returncode == 0:
        for raw in sorted(set(changed.stdout.split(b"\0")) - {b""}):
            relative = os.fsdecode(raw)
            path = REPO / relative
            files[relative] = sha256(path) if path.is_file() else None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
        "changed_source_sha256": files,
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }


def new_artifact(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 2,
        "kind": args.command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": vars(args),
        "environment": environment(),
        "results": [],
        "errors": [],
    }


def selected_cases(args: argparse.Namespace) -> list[dict]:
    source = read_json(args.cases)
    identity = model_identity(args.model)
    for key in ("model_config_sha256", "tokenizer_sha256"):
        if source[key] != identity[key]:
            raise ValueError(f"Case file {key} differs from the selected model")
    cases = source["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        found = {case["case_id"] for case in cases}
        if wanted - found:
            raise ValueError(f"Unknown case IDs: {sorted(wanted - found)}")
        cases = [case for case in cases if case["case_id"] in wanted]
    else:
        suite = "validation" if args.suite == "smoke" else args.suite
        cases = [case for case in cases if case["suite"] == suite]
        if args.suite == "smoke":
            cases = cases[:1]
    if not cases:
        raise ValueError("No selected input cases")
    for case in cases:
        if len(case["input_ids"]) != case["input_tokens"] or case["max_new_tokens"] < 2:
            raise ValueError(f"Invalid input or output length for {case['case_id']}")
    return cases


def prepare(args: argparse.Namespace, artifact: dict) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    configurations = [
        ("validation", "A17", 17, 16, "The capital of France is"),
        ("validation", "B129", 129, 16, "The sum of two and three is"),
        ("validation", "C513", 513, 16, "The largest planet in the solar system is"),
    ]
    lengths = [int(value) for value in args.input_lengths.split(",")]
    if (
        not lengths
        or len(set(lengths)) != len(lengths)
        or min(lengths) < 16
        or args.output_tokens < 2
    ):
        raise ValueError(
            "Input lengths must be distinct and >=16; output length must be >=2"
        )
    configurations.extend(
        (
            "benchmark",
            f"bench-{length}-{args.output_tokens}",
            length,
            args.output_tokens,
            "Continue this passage:",
        )
        for length in lengths
    )
    filler = tokenizer.encode(
        " A computer processes instructions and stores intermediate results in memory.",
        add_special_tokens=False,
    )
    cases = []
    for suite, case_id, length, outputs, question in configurations:
        suffix = tokenizer.encode("\n" + question, add_special_tokens=False)
        fill_length = length - len(suffix)
        if fill_length <= 0 or not filler:
            raise ValueError("Prompt cannot fit the requested input length")
        ids = (filler * ((fill_length + len(filler) - 1) // len(filler)))[
            :fill_length
        ] + suffix
        cases.append(
            {
                "case_id": case_id,
                "suite": suite,
                "input_tokens": len(ids),
                "max_new_tokens": outputs,
                "input_ids": ids,
                "decoded_input": tokenizer.decode(ids),
            }
        )
    identity = model_identity(args.model)
    artifact.update(
        model=args.model,
        model_identity=identity,
        **identity,
        cases=cases,
        prompt_format="raw text; no chat template or implicit special tokens",
    )


def native_engine_kwargs(args: argparse.Namespace, *, output_logprobs: bool) -> dict:
    return {
        "model": args.model,
        "device": "npu",
        "base_gpu_id": 0,
        "world_size": args.tp_size,
        "nprocs_per_node": args.tp_size,
        "attn_tp_size": args.tp_size,
        "dense_tp_size": args.tp_size,
        "lm_head_tp_size": args.tp_size,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "dtype": "bfloat16",
        "mamba_ssm_dtype": "float32",
        "language_model_only": True,
        "trust_remote_code": False,
        "retraction_snapshot_ratio": 0.0,
        "attention_backend": args.attention_backend,
        "sampling_backend": args.sampling_backend,
        "logprob_order": "torch",
        "enforce_eager": True,
        "disable_prefill_graph": True,
        "disable_pdl": True,
        "disable_autotune": True,
        "disable_overlap_schedule": args.overlap_schedule == "off",
        "enable_prefix_caching": False,
        "disable_kvstore": True,
        "enable_output_logprobs": output_logprobs,
        "max_num_seqs": args.concurrency,
        "max_model_len": args.max_model_len,
        "max_total_tokens": args.max_total_tokens,
        "max_prefill_tokens": args.prefill_chunk_tokens,
        "chunked_prefill_size": args.prefill_chunk_tokens,
        "gpu_memory_utilization": args.memory_utilization,
        "stream_interval": 1,
        "stream_output": True,
        "seed": 0,
        "log_level": "info",
        "enable_log_requests": False,
        "distributed_timeout_seconds": args.distributed_timeout_seconds,
    }


def run_request(
    engine,
    case: dict,
    *,
    request_id: str,
    phase: str,
    run_id: str,
    group_id: str,
    barrier: threading.Barrier | None,
) -> dict:
    if barrier is not None:
        barrier.wait(timeout=60)
    started = time.perf_counter()
    collector = StreamRecord(start_s=started)
    for frame in engine.generate(
        input_ids=case["input_ids"],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": case["max_new_tokens"],
            "ignore_eos": True,
        },
        return_logprob=phase == "validation",
        stream=True,
    ):
        collector.add(frame, arrival_s=time.perf_counter())
    ended = time.perf_counter()
    row = {
        **case,
        **collector.finish(end_s=ended),
        "phase": phase,
        "run_id": run_id,
        "request_id": request_id,
        "group_id": group_id,
        "request_start_monotonic_s": started,
        "request_end_monotonic_s": ended,
    }
    row["requested_output_count_met"] = (
        row["actual_output_tokens"] == case["max_new_tokens"]
    )
    if not row["requested_output_count_met"]:
        raise ValueError(f"Wrong output count in {request_id}")
    if (
        phase == "validation"
        and len(row["token_logprobs"]) != row["actual_output_tokens"]
    ):
        raise ValueError(f"Missing output logprobs in {request_id}")
    return row


def run_group(
    engine, case: dict, *, concurrency: int, phase: str, group_id: str
) -> tuple[list[dict], dict]:
    starts = []
    barrier = threading.Barrier(
        concurrency + 1, action=lambda: starts.append(time.perf_counter())
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_request,
                engine,
                case,
                request_id=f"{group_id}-request-{index}",
                phase=phase,
                run_id=f"{group_id}-request-{index}",
                group_id=group_id,
                barrier=barrier,
            )
            for index in range(concurrency)
        ]
        barrier.wait(timeout=60)
        rows = [future.result() for future in futures]
    group = concurrent_group(
        rows,
        start_s=starts[0],
        end_s=max(row["request_end_monotonic_s"] for row in rows),
    )
    group.update(case_id=case["case_id"], group_id=group_id, phase=phase)
    return rows, group


def native(args: argparse.Namespace, artifact: dict) -> None:
    chips = visible_chips(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""), expected=args.tp_size
    )
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.entrypoints.engine import Engine

    source = Path(sys.modules[Engine.__module__].__file__).resolve()
    if source != REPO / "python/tokenspeed/runtime/entrypoints/engine.py":
        raise ValueError(f"Engine is not imported from this checkout: {source}")
    if args.suite == "validation" and args.concurrency != 1:
        raise ValueError(
            "Sequential state-isolation validation requires --concurrency 1"
        )
    cases = selected_cases(args)
    if args.suite == "smoke":
        cases = [{**case, "max_new_tokens": 4} for case in cases]
    if any(
        case["input_tokens"] + case["max_new_tokens"] > args.max_model_len
        for case in cases
    ):
        raise ValueError("Input plus output exceeds model context limit")
    output = Path(args.output).resolve()
    evidence_dir = output.parent / f"{output.stem}.workers-{uuid.uuid4().hex[:8]}"
    evidence_dir.mkdir(parents=True)
    worker_env = {
        "AICS_TP_EVIDENCE_DIR": str(evidence_dir),
        "AICS_TP_MEMORY_INTERVAL_S": str(args.memory_interval_s),
        "AICS_TP_RECORD_COLLECTIVES": args.record_collectives,
    }
    previous = {name: os.environ.get(name) for name in worker_env}
    os.environ.update(worker_env)
    engine_args = native_engine_kwargs(args, output_logprobs=args.suite == "validation")
    artifact.update(
        cases_sha256=sha256(args.cases),
        model_identity=model_identity(args.model),
        engine_args=engine_args,
        evidence_directory=str(evidence_dir),
        device_mapping=[
            {"rank": rank, "logical_device": rank, "physical_chip": chip}
            for rank, chip in enumerate(chips)
        ],
        timing_scope="Host-observed streaming, including scheduler, IPC and delivery; unprofiled; memory and forward evidence collection enabled",
        groups=[],
    )
    write_json(args.output, artifact)
    engine = None
    startup = time.perf_counter()
    try:
        engine = Engine(**engine_args)
        artifact["engine_startup_s"] = time.perf_counter() - startup
        if args.suite == "validation":
            if len(cases) < 2:
                raise ValueError("Validation requires at least two distinct cases")
            first, second = cases[:2]
            jobs = [(first, "A-before"), (second, "B"), (first, "A-after")]
            jobs.extend((case, case["case_id"]) for case in cases[2:])
            for case, run_id in jobs:
                row = run_request(
                    engine,
                    case,
                    request_id=run_id,
                    run_id=run_id,
                    group_id=run_id,
                    phase="validation",
                    barrier=None,
                )
                row["worker_memory"] = summarize_memory(
                    evidence_dir,
                    start_s=row["request_start_monotonic_s"],
                    end_s=row["request_end_monotonic_s"],
                )
                artifact["results"].append(row)
                write_json(args.output, artifact)
                print(
                    f"TP{args.tp_size} {run_id}: {row['actual_output_tokens']} output tokens",
                    flush=True,
                )
            before, after = artifact["results"][0], artifact["results"][2]
            artifact["request_isolation"] = {
                "sequence": [row["case_id"] for row in artifact["results"]],
                "same_token_ids": before["output_ids"] == after["output_ids"],
                "same_token_logprobs": before["token_logprobs"]
                == after["token_logprobs"],
                "logprobs_complete": len(before["token_logprobs"])
                == before["actual_output_tokens"]
                and len(after["token_logprobs"]) == after["actual_output_tokens"],
                "intervening_distinct_request": first["input_ids"]
                != second["input_ids"],
                "cache_flush_between_requests": False,
            }
        else:
            for case in cases:
                for index in range(args.warmup + args.repeats):
                    phase = "warmup" if index < args.warmup else "measurement"
                    group_id = f"{case['case_id']}-{phase}-{index}"
                    rows, group = run_group(
                        engine,
                        case,
                        concurrency=args.concurrency,
                        phase=phase,
                        group_id=group_id,
                    )
                    group["worker_memory"] = summarize_memory(
                        evidence_dir,
                        start_s=group["start_monotonic_s"],
                        end_s=group["end_monotonic_s"],
                    )
                    artifact["results"].extend(rows)
                    artifact["groups"].append(group)
                    write_json(args.output, artifact)
                    print(
                        f"TP{args.tp_size} {group_id}: C={args.concurrency}, {group['output_tokens_per_s']:.3f} output tokens/s",
                        flush=True,
                    )
            artifact["benchmark_summary"] = benchmark_summary(
                artifact["results"], artifact["groups"]
            )
    finally:
        artifact["worker_memory_all"] = summarize_memory(
            evidence_dir, start_s=startup, end_s=time.perf_counter()
        )
        if engine is not None:
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
        raise RuntimeError(
            f"Incomplete worker execution evidence: {artifact['worker_evidence']}"
        )
    if args.suite == "validation" and not all(
        artifact["request_isolation"][key]
        for key in (
            "same_token_ids",
            "same_token_logprobs",
            "logprobs_complete",
            "intervening_distinct_request",
        )
    ):
        raise RuntimeError("Repeated request results or state isolation differ")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--input-lengths", default="128,512")
    prepare_parser.add_argument("--output-tokens", type=int, default=64)
    prepare_parser.add_argument("--output", required=True)
    run_parser = commands.add_parser("native")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--cases", required=True)
    run_parser.add_argument("--case-ids")
    run_parser.add_argument(
        "--suite", choices=("smoke", "validation", "benchmark"), required=True
    )
    run_parser.add_argument("--tp-size", type=int, choices=(1, 2), required=True)
    run_parser.add_argument("--concurrency", type=int, required=True)
    run_parser.add_argument("--attention-backend", required=True)
    run_parser.add_argument("--sampling-backend", required=True)
    run_parser.add_argument("--overlap-schedule", choices=("on", "off"), required=True)
    run_parser.add_argument(
        "--record-collectives", choices=("on", "off"), required=True
    )
    run_parser.add_argument("--prefill-chunk-tokens", type=int, required=True)
    run_parser.add_argument("--memory-utilization", type=float, required=True)
    run_parser.add_argument("--max-model-len", type=int, default=1024)
    run_parser.add_argument("--max-total-tokens", type=int, default=16384)
    run_parser.add_argument("--memory-interval-s", type=float, default=0.2)
    run_parser.add_argument("--distributed-timeout-seconds", type=int, default=180)
    run_parser.add_argument("--warmup", type=int, default=1)
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = parser().parse_args()
    artifact = new_artifact(args)
    write_json(args.output, artifact)
    try:
        if args.command == "native":
            if (
                args.concurrency < 1
                or args.warmup < 0
                or args.repeats < 1
                or args.prefill_chunk_tokens < 1
            ):
                raise ValueError(
                    "Concurrency, chunk size and repeats must be positive; warmup cannot be negative"
                )
            if not 0 < args.memory_utilization < 1 or args.memory_interval_s <= 0:
                raise ValueError(
                    "Memory utilization must be in (0,1) and sampling interval positive"
                )
            if (
                args.max_model_len < 1
                or args.max_total_tokens < 1
                or args.distributed_timeout_seconds < 1
            ):
                raise ValueError(
                    "Context, cache capacity and distributed timeout must be positive"
                )
        {"prepare": prepare, "native": native}[args.command](args, artifact)
        artifact["status"] = "completed"
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["errors"].append(
            {"exception": repr(exc), "traceback": traceback.format_exc()}
        )
        traceback.print_exc()
    finally:
        artifact["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, artifact)
    return 0 if artifact["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
