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

"""Reproducible Qwen3.5 reference, native validation, and single-chip benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from metrics import StreamRecord, compare_sequences
from precision import (
    collect_worker_precision,
    configure_npu_precision,
    install_spawned_worker_hooks,
    read_npu_precision,
)
from worker_memory import start_worker_sampler, summarize_memory

MODEL = os.environ.get("AICS_MODEL_PATH", "models/Qwen3.5-0.8B")
REPO = Path(__file__).resolve().parents[2]

# Only install during spawn bootstrap; actual device configuration and sampling
# start when the model worker enters BaseProcess.run, before model loading.
if __name__ == "__mp_main__":
    # Target unpickling imports platform dispatch: register the lazy NPU module,
    # but do not set ACL options or initialize its compiler during bootstrap.
    import torch_npu  # noqa: F401

    install_spawned_worker_hooks(start_worker_sampler)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary.replace(destination)


def json_safe(value):
    """Retain nonfinite numerical failures explicitly in standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment() -> dict:
    versions = {}
    for name in (
        "torch",
        "torch-npu",
        "transformers",
        "tokenspeed",
        "tokenspeed-scheduler",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
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
    source_files = {}
    if changed.returncode == 0:
        for raw in sorted(set(changed.stdout.split(b"\0")) - {b""}):
            relative = os.fsdecode(raw)
            path = REPO / relative
            source_files[relative] = sha256(path) if path.is_file() else None
    return {
        "python": sys.version,
        "changed_source_sha256": source_files,
        "platform": platform.platform(),
        "versions": versions,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }


def new_artifact(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 1,
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
    if source["model"] != args.model:
        raise ValueError("Cases were prepared for a different model path")
    if source["model_config_sha256"] != sha256(Path(args.model) / "config.json"):
        raise ValueError("Model configuration changed since cases were prepared")
    if source["tokenizer_sha256"] != sha256(Path(args.model) / "tokenizer.json"):
        raise ValueError("Tokenizer changed since cases were prepared")
    cases = source["cases"]
    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        found = {case["case_id"] for case in cases}
        if wanted - found:
            raise ValueError(f"Unknown case IDs: {sorted(wanted - found)}")
        cases = [case for case in cases if case["case_id"] in wanted]
    else:
        cases = [case for case in cases if case["suite"] == args.suite]
    if not cases:
        raise ValueError("No cases selected")
    for case in cases:
        if len(case["input_ids"]) != case["input_tokens"]:
            raise ValueError(f"Wrong input length in {case['case_id']}")
    return cases


def prepare(args: argparse.Namespace, artifact: dict) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = []
    configurations = [
        ("validation", "A17", 17, 16, "The capital of France is"),
        ("validation", "B129", 129, 16, "The sum of two and three is"),
        ("validation", "C513", 513, 16, "The largest planet in the solar system is"),
    ]
    configurations.extend(
        (
            "benchmark",
            f"bench-{inputs}-{outputs}",
            inputs,
            outputs,
            "Continue this passage:",
        )
        for inputs in (128, 512)
        for outputs in (16, 64)
    )
    filler = tokenizer.encode(
        " A computer processes instructions and stores intermediate results in memory.",
        add_special_tokens=False,
    )
    for suite, case_id, length, outputs, question in configurations:
        suffix = tokenizer.encode("\n" + question, add_special_tokens=False)
        if len(suffix) >= length or not filler:
            raise ValueError(
                "Tokenized prompt construction does not fit requested length"
            )
        fill_length = length - len(suffix)
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
    artifact.update(
        {
            "model": args.model,
            "model_config_sha256": sha256(Path(args.model) / "config.json"),
            "tokenizer_sha256": sha256(Path(args.model) / "tokenizer.json"),
            "prompt_format": "raw text, no chat template, no implicit special tokens",
            "cases": cases,
        }
    )


def reference(args: argparse.Namespace, artifact: dict) -> None:
    import torch

    if args.device == "npu":
        import torch_npu  # noqa: F401

        require_one_chip()
        artifact["npu_precision"] = configure_npu_precision()
        torch.npu.set_device(0)
    from transformers import Qwen3_5ForConditionalGeneration

    cases = selected_cases(args)
    artifact["cases_sha256"] = sha256(args.cases)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    artifact["reference_class"] = f"{type(model).__module__}.{type(model).__name__}"
    artifact["attention_implementation"] = "eager"
    artifact["model_config_sha256"] = sha256(Path(args.model) / "config.json")
    jobs = [(case, None) for case in cases]
    if args.native_trajectories:
        native = read_json(args.native_trajectories)
        by_case = {case["case_id"]: case for case in cases}
        jobs = [
            (by_case[row["case_id"]], row)
            for row in native["results"]
            if row["case_id"] in by_case and row["phase"] == "validation"
        ]
        if not jobs:
            raise ValueError("No selected native validation trajectories")
    with torch.inference_mode():
        for case, native_row in jobs:
            ids = list(case["input_ids"])
            steps = (
                len(native_row["output_ids"]) if native_row else case["max_new_tokens"]
            )
            generated, greedy, logprobs, margins = [], [], [], []
            cache = None
            started = time.perf_counter()
            for step in range(steps):
                full_ids = ids + generated
                current_ids = (
                    full_ids
                    if args.cache_mode == "recompute" or step == 0
                    else full_ids[-1:]
                )
                inputs = torch.tensor(
                    [current_ids], dtype=torch.long, device=args.device
                )
                mask = torch.ones(
                    (1, len(full_ids)), dtype=torch.long, device=args.device
                )
                output = model(
                    input_ids=inputs,
                    attention_mask=mask,
                    past_key_values=cache if args.cache_mode == "cached" else None,
                    use_cache=args.cache_mode == "cached",
                    logits_to_keep=1,
                )
                if args.cache_mode == "cached":
                    cache = output.past_key_values
                logits = output.logits[0, -1].float()
                values, _ = torch.topk(logits, 2)
                greedy_id = int(torch.argmax(logits).item())
                chosen = (
                    int(native_row["output_ids"][step]) if native_row else greedy_id
                )
                greedy.append(greedy_id)
                generated.append(chosen)
                logprobs.append(float(torch.log_softmax(logits, -1)[chosen].item()))
                margins.append(float((values[0] - values[1]).item()))
            row = {
                **case,
                "run_id": native_row["run_id"] if native_row else case["case_id"],
                "trajectory": "native_teacher_forced" if native_row else "greedy",
                "output_ids": generated,
                "actual_output_tokens": len(generated),
                "greedy_ids": greedy,
                "token_logprobs": logprobs,
                "top2_logit_margins": margins,
                "reference_elapsed_s": time.perf_counter() - started,
                "eos_policy": "ignore EOS; exact requested step count",
            }
            artifact["results"].append(row)
            write_json(args.output, artifact)
            print(
                f"Reference {row['run_id']}: {len(ids)} input, {len(generated)} output tokens",
                flush=True,
            )
    if args.device == "npu":
        artifact["npu_precision_after"] = read_npu_precision()
        if not artifact["npu_precision_after"]["matches_requested"]:
            raise RuntimeError("Reference changed NPU precision during inference")


def require_one_chip() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible.isdecimal():
        raise ValueError(
            "Set ASCEND_RT_VISIBLE_DEVICES to one assigned physical chip ID"
        )


def native_engine_kwargs(args: argparse.Namespace, *, output_logprobs: bool) -> dict:
    """Keep validation, timing and separate profiling engine settings aligned."""
    return {
        "model": args.model,
        "device": "npu",
        "base_gpu_id": 0,
        "dtype": "bfloat16",
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
        "disable_overlap_schedule": True,
        "enable_prefix_caching": False,
        "disable_kvstore": True,
        "enable_output_logprobs": output_logprobs,
        "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "max_total_tokens": args.max_total_tokens,
        "max_prefill_tokens": 512,
        "chunked_prefill_size": 512,
        "gpu_memory_utilization": args.memory_utilization,
        "stream_interval": 1,
        "stream_output": True,
        "seed": 0,
        "log_level": "info",
        "enable_log_requests": False,
    }


def native(args: argparse.Namespace, artifact: dict) -> None:
    require_one_chip()
    import torch_npu  # noqa: F401: register NPU before importing TokenSpeed.

    artifact["npu_precision"] = configure_npu_precision()

    from tokenspeed.runtime.entrypoints.engine import Engine

    engine_source = Path(sys.modules[Engine.__module__].__file__).resolve()
    if engine_source != REPO / "python/tokenspeed/runtime/entrypoints/engine.py":
        raise ValueError(
            f"Expected Engine from this checkout; imported {engine_source}. Set source PYTHONPATH."
        )
    cases = selected_cases(args)
    artifact["cases_sha256"] = sha256(args.cases)
    if any(
        case["input_tokens"] + case["max_new_tokens"] > args.max_model_len
        for case in cases
    ):
        raise ValueError("Case exceeds configured model context length")
    memory_dir = (
        Path(args.output).resolve().parent
        / f"{Path(args.output).stem}.memory-{uuid.uuid4().hex[:8]}"
    )
    memory_dir.mkdir(parents=True)
    os.environ["AICS_WORKER_MEMORY_DIR"] = str(memory_dir)
    os.environ["AICS_NPU_PRECISION_DIR"] = str(memory_dir)
    os.environ["AICS_WORKER_MEMORY_INTERVAL_S"] = str(args.memory_interval_s)
    artifact["memory_samples_dir"] = str(memory_dir)
    artifact["timing_scope"] = (
        "Host-observed Engine.generate streaming: includes frontend, scheduler, IPC and delivery; "
        "not isolated prefill/decode kernel timing. Worker memory sampling enabled."
    )
    engine_args = native_engine_kwargs(args, output_logprobs=args.suite == "validation")
    artifact["engine_args"] = engine_args
    artifact["engine_module"] = sys.modules[Engine.__module__].__file__
    write_json(args.output, artifact)
    engine = None
    startup_begin = time.perf_counter()
    try:
        engine = Engine(**engine_args)
        artifact["engine_startup_s"] = time.perf_counter() - startup_begin
        jobs = []
        if args.suite == "validation":
            # Deliberately reuse the same one-slot engine without flush/reset.
            if len(cases) >= 2:
                first, second = cases[:2]
                jobs = [
                    (first, "validation", "A-before"),
                    (second, "validation", "B"),
                    (first, "validation", "A-after"),
                ]
                jobs.extend((case, "validation", case["case_id"]) for case in cases[2:])
            else:
                jobs = [
                    (cases[0], "validation", "A-before"),
                    (cases[0], "validation", "A-after"),
                ]
        else:
            for case in cases:
                jobs.extend(
                    (case, "warmup", f"{case['case_id']}-warmup-{i}")
                    for i in range(args.warmup)
                )
                jobs.extend(
                    (case, "measurement", f"{case['case_id']}-repeat-{i}")
                    for i in range(args.repeats)
                )
        for case, phase, run_id in jobs:
            started = time.perf_counter()
            collector = StreamRecord(start_s=started)
            for frame in engine.generate(
                input_ids=case["input_ids"],
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": case["max_new_tokens"],
                    "ignore_eos": True,
                },
                return_logprob=args.suite == "validation",
                stream=True,
            ):
                collector.add(frame, arrival_s=time.perf_counter())
            ended = time.perf_counter()
            row = {
                **case,
                **collector.finish(end_s=ended),
                "phase": phase,
                "run_id": run_id,
            }
            row["request_start_monotonic_s"] = started
            row["request_end_monotonic_s"] = ended
            row["worker_memory"] = summarize_memory(
                memory_dir, start_s=started, end_s=ended
            )
            row["requested_output_count_met"] = (
                row["actual_output_tokens"] == case["max_new_tokens"]
            )
            artifact["results"].append(row)
            write_json(args.output, artifact)
            print(
                f"Native {run_id}: {row['actual_output_tokens']} tokens, TTFT={row['observed_ttft_s']}, TPOT={row['observed_tpot_s']}",
                flush=True,
            )
            if not row["requested_output_count_met"]:
                raise ValueError(f"Unexpected early termination for {run_id}")
            if (
                args.suite == "validation"
                and len(row["token_logprobs"]) != row["actual_output_tokens"]
            ):
                raise ValueError(f"Incomplete output logprobs for {run_id}")
        if args.suite == "validation":
            before = artifact["results"][0]
            after = next(
                row for row in artifact["results"] if row["run_id"] == "A-after"
            )
            artifact["request_isolation"] = {
                "sequence": [row["case_id"] for row in artifact["results"][:3]],
                "same_token_ids": before["output_ids"] == after["output_ids"],
                "same_token_logprobs": before["token_logprobs"]
                == after["token_logprobs"],
                "logprobs_complete": len(before["token_logprobs"])
                == before["actual_output_tokens"]
                and len(after["token_logprobs"]) == after["actual_output_tokens"],
                "intervening_distinct_request": len(cases) >= 2,
                "cache_flush_between_requests": False,
            }
        else:
            summaries = []
            for case in cases:
                rows = [
                    row
                    for row in artifact["results"]
                    if row["case_id"] == case["case_id"]
                    and row["phase"] == "measurement"
                ]
                summary = {
                    "case_id": case["case_id"],
                    "input_tokens": case["input_tokens"],
                    "requested_output_tokens": case["max_new_tokens"],
                    "repeats": len(rows),
                }
                for key in ("observed_ttft_s", "observed_tpot_s", "request_duration_s"):
                    values = [row[key] for row in rows if row[key] is not None]
                    summary[key] = (
                        {
                            "median": statistics.median(values),
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                        }
                        if values
                        else None
                    )
                summaries.append(summary)
            artifact["benchmark_summary"] = summaries
    finally:
        artifact["worker_memory_all"] = summarize_memory(
            memory_dir, start_s=startup_begin, end_s=time.perf_counter()
        )
        if engine is not None:
            engine.shutdown()
        artifact["worker_precision"] = collect_worker_precision(memory_dir)
        os.environ.pop("AICS_WORKER_MEMORY_DIR", None)
        os.environ.pop("AICS_WORKER_MEMORY_INTERVAL_S", None)
        os.environ.pop("AICS_NPU_PRECISION_DIR", None)


def compare(args: argparse.Namespace, artifact: dict) -> None:
    native_data, ref_data = read_json(args.native), read_json(args.reference)
    if native_data["status"] != "completed" or ref_data["status"] != "completed":
        raise ValueError("Compare requires completed artifacts")
    if native_data["cases_sha256"] != ref_data["cases_sha256"]:
        raise ValueError("Artifacts used different case files")
    for row in native_data["results"]:
        if row["phase"] != "validation":
            continue
        candidates = [
            ref
            for ref in ref_data["results"]
            if ref["case_id"] == row["case_id"]
            and (ref["trajectory"] == "greedy" or ref["run_id"] == row["run_id"])
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one reference for {row['run_id']}, got {len(candidates)}"
            )
        artifact["results"].append(
            compare_sequences(row, candidates[0], logprob_atol=args.logprob_atol)
        )
    if not artifact["results"]:
        raise ValueError("No validation rows to compare")
    isolation = native_data["request_isolation"]
    artifact["request_isolation"] = isolation
    artifact["strict_pass"] = all(
        row["strict_pass"] for row in artifact["results"]
    ) and all(
        isolation[key]
        for key in (
            "same_token_ids",
            "same_token_logprobs",
            "logprobs_complete",
            "intervening_distinct_request",
        )
    )
    artifact["limitations"] = (
        "This checks the selected prompts and generated trajectories, not arbitrary inputs. "
        "Greedy disagreements near a small reference top-2 margin need numerical investigation; "
        "teacher-forced scoring distinguishes changed history from numerical error."
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "reference", "native", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True)
        if name != "compare":
            command.add_argument("--model", default=MODEL)
        if name in ("reference", "native"):
            command.add_argument("--cases", required=True)
            command.add_argument(
                "--suite", choices=("validation", "benchmark"), required=True
            )
            command.add_argument("--case-ids", help="Optional comma-separated case IDs")
        if name == "reference":
            command.add_argument("--device", choices=("cpu", "npu"), required=True)
            command.add_argument(
                "--dtype", choices=("float32", "bfloat16"), required=True
            )
            command.add_argument(
                "--cache-mode", choices=("cached", "recompute"), required=True
            )
            command.add_argument("--cpu-threads", type=int, default=4)
            command.add_argument(
                "--native-trajectories",
                help="Score native IDs instead of generating the reference's own trajectory",
            )
        if name == "native":
            command.add_argument("--attention-backend", required=True)
            command.add_argument("--sampling-backend", required=True)
            command.add_argument("--max-model-len", type=int, default=1024)
            command.add_argument("--max-total-tokens", type=int, default=16384)
            command.add_argument("--memory-utilization", type=float, default=0.15)
            command.add_argument("--memory-interval-s", type=float, default=0.2)
            command.add_argument("--warmup", type=int, default=1)
            command.add_argument("--repeats", type=int, default=3)
        if name == "compare":
            command.add_argument("--native", required=True)
            command.add_argument("--reference", required=True)
            command.add_argument("--logprob-atol", type=float, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    artifact = new_artifact(args)
    write_json(args.output, artifact)
    try:
        if args.command == "native" and (
            args.warmup < 0 or args.repeats < 1 or args.memory_interval_s <= 0
        ):
            raise ValueError(
                "Need nonnegative warmup and positive repeats/memory interval"
            )
        if args.command == "compare" and args.logprob_atol < 0:
            raise ValueError("logprob-atol must be nonnegative")
        {
            "prepare": prepare,
            "reference": reference,
            "native": native,
            "compare": compare,
        }[args.command](args, artifact)
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
    if artifact["status"] != "completed":
        return 1
    if args.command == "compare" and not artifact["strict_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
