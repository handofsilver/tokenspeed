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

"""Compare TP2 output with TP1 or same-history Transformers output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from examples.aics_qwen35.metrics import compare_sequences

IDENTITY_FIELDS = (
    "model_config_sha256",
    "tokenizer_sha256",
    "weight_index_sha256",
    "weight_files",
)
CRITICAL_ENGINE_FIELDS = (
    "dtype",
    "mamba_ssm_dtype",
    "language_model_only",
    "trust_remote_code",
    "retraction_snapshot_ratio",
    "attention_backend",
    "sampling_backend",
    "logprob_order",
    "enforce_eager",
    "disable_prefill_graph",
    "disable_pdl",
    "disable_autotune",
    "disable_overlap_schedule",
    "enable_prefix_caching",
    "disable_kvstore",
    "enable_output_logprobs",
    "max_num_seqs",
    "max_model_len",
    "max_total_tokens",
    "max_prefill_tokens",
    "chunked_prefill_size",
    "stream_interval",
    "stream_output",
    "seed",
    "data_parallel_size",
    "pipeline_parallel_size",
)
IGNORED_ENGINE_FIELDS = {
    "model",
    "world_size",
    "nprocs_per_node",
    "attn_tp_size",
    "dense_tp_size",
    "lm_head_tp_size",
    "base_gpu_id",
    "gpu_memory_utilization",
    "distributed_timeout_seconds",
    "log_level",
    "enable_log_requests",
}
ISOLATION_FIELDS = (
    "same_token_ids",
    "same_token_logprobs",
    "logprobs_complete",
    "intervening_distinct_request",
)


def required(mapping: dict, key: str, context: str):
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"Missing {context}.{key}")
    return mapping[key]


def validate_native(artifact: dict) -> tuple[dict, dict, int]:
    if artifact.get("kind") != "native":
        raise ValueError("Expected a native inference artifact")
    identity = required(artifact, "model_identity", "native")
    for field in IDENTITY_FIELDS:
        if field not in identity:
            raise ValueError(f"Missing native.model_identity.{field}")
    for field in ("model_config_sha256", "tokenizer_sha256"):
        if not isinstance(identity[field], str) or not identity[field]:
            raise ValueError(f"Invalid native model identity: {field}")
    files = identity["weight_files"]
    if not files or any(
        not row.get("name")
        or not isinstance(row.get("bytes"), int)
        or row["bytes"] <= 0
        for row in files
    ):
        raise ValueError("Missing or invalid native weight-file metadata")
    if len({row["name"] for row in files}) != len(files):
        raise ValueError("Duplicate native weight-file metadata")
    engine = required(artifact, "engine_args", "native")
    for field in CRITICAL_ENGINE_FIELDS:
        required(engine, field, "native.engine_args")
    if required(engine, "device", "native.engine_args") != "npu":
        raise ValueError("Expected a native NPU run")
    tp_size = required(artifact["config"], "tp_size", "native.config")
    if tp_size not in (1, 2):
        raise ValueError("Expected TP size 1 or 2")
    for field in (
        "world_size",
        "nprocs_per_node",
        "attn_tp_size",
        "dense_tp_size",
        "lm_head_tp_size",
    ):
        if required(engine, field, "native.engine_args") != tp_size:
            raise ValueError(f"Inconsistent TP dimension: {field}")
    if engine["data_parallel_size"] != 1 or engine["pipeline_parallel_size"] != 1:
        raise ValueError("Comparison requires tensor parallelism with DP=PP=1")
    return identity, engine, tp_size


def pairing_evidence(native: dict, reference: dict) -> dict:
    identity, engine, native_tp = validate_native(native)
    if reference.get("kind") == "native":
        other_identity, other_engine, reference_tp = validate_native(reference)
        for field in IDENTITY_FIELDS:
            left, right = identity[field], other_identity[field]
            if field == "weight_files":
                left = sorted(left, key=lambda row: row["name"])
                right = sorted(right, key=lambda row: row["name"])
            if left != right:
                raise ValueError(f"Runs used different model identity: {field}")
        checked = (set(engine) | set(other_engine)) - IGNORED_ENGINE_FIELDS
        for field in sorted(checked):
            if (
                field not in engine
                or field not in other_engine
                or engine[field] != other_engine[field]
            ):
                raise ValueError(f"Runs used different Engine configuration: {field}")
        return {
            "reference_kind": "native",
            "native_tp_size": native_tp,
            "reference_tp_size": reference_tp,
            "model_identity_fields_checked": list(IDENTITY_FIELDS),
            "engine_fields_checked": sorted(checked),
            "unverified_identity_fields": ["weight_file_content_sha256"],
            "identity_scope": "config, tokenizer, weight index and weight filenames/sizes; full weight checksums are external download evidence",
            "dtype": engine["dtype"],
        }
    if reference.get("kind") != "reference":
        raise ValueError("Expected a native run or a Transformers reference artifact")
    reference_config = required(reference, "config", "reference")
    dtype = required(reference_config, "dtype", "reference.config")
    if engine["dtype"] != dtype:
        raise ValueError("Runs used different weight precision")
    config_hash = required(reference, "model_config_sha256", "reference")
    if identity["model_config_sha256"] != config_hash:
        raise ValueError("Runs used different model configuration")
    if not required(reference, "reference_class", "reference").endswith(
        ".Qwen3_5ForConditionalGeneration"
    ):
        raise ValueError("Unrecognized Transformers reference class")
    if required(reference, "attention_implementation", "reference") != "eager":
        raise ValueError("Expected the declared eager Transformers reference")
    cache_mode = required(reference_config, "cache_mode", "reference.config")
    if cache_mode not in ("cached", "recompute"):
        raise ValueError("Unknown reference cache mode")
    return {
        "reference_kind": "transformers",
        "native_tp_size": native_tp,
        "model_identity_fields_checked": ["model_config_sha256"],
        "unverified_identity_fields": [
            "tokenizer_sha256",
            "weight_index_sha256",
            "weight_files",
            "weight_file_content_sha256",
        ],
        "identity_scope": "configuration only; the legacy reference artifact does not record tokenizer or weight identity",
        "dtype": dtype,
        "reference_cache_mode": cache_mode,
        "engine_fields_checked": [],
    }


def validation_rows(artifact: dict, *, native: bool) -> list[dict]:
    rows = (
        [row for row in artifact["results"] if row.get("phase") == "validation"]
        if native
        else artifact["results"]
    )
    if not rows:
        raise ValueError("No validation requests")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate validation run IDs")
    for row in rows:
        expected = required(row, "max_new_tokens", "request")
        if not isinstance(expected, int) or expected <= 0:
            raise ValueError("Invalid requested output count")
        if (
            len(row["output_ids"]) != expected
            or required(row, "actual_output_tokens", "request") != expected
        ):
            raise ValueError(f"Wrong output count for {row['run_id']}")
        if required(row, "input_tokens", "request") != len(row["input_ids"]):
            raise ValueError(f"Wrong input count for {row['run_id']}")
        if any(not math.isfinite(float(value)) for value in row["token_logprobs"]):
            raise ValueError(f"Nonfinite token logprob for {row['run_id']}")
    return rows


def isolation_evidence(artifact: dict, rows: list[dict]) -> dict:
    by_id = {row["run_id"]: row for row in rows}
    if not {"A-before", "B", "A-after"} <= set(by_id):
        raise ValueError("State-isolation validation requires A-before, B, A-after")
    before, middle, after = (by_id[name] for name in ("A-before", "B", "A-after"))
    positions = {row["run_id"]: index for index, row in enumerate(rows)}
    actual = {
        "same_token_ids": before["input_ids"] == after["input_ids"]
        and before["output_ids"] == after["output_ids"],
        "same_token_logprobs": before["token_logprobs"] == after["token_logprobs"],
        "logprobs_complete": all(
            len(row["token_logprobs"]) == len(row["output_ids"])
            and all(math.isfinite(float(value)) for value in row["token_logprobs"])
            for row in (before, after)
        ),
        "intervening_distinct_request": positions["A-before"]
        < positions["B"]
        < positions["A-after"]
        and middle["input_ids"] != before["input_ids"],
    }
    claimed = required(artifact, "request_isolation", "native")
    passed = (
        all(actual.values())
        and all(claimed.get(key) is True for key in ISOLATION_FIELDS)
        and claimed.get("cache_flush_between_requests") is False
    )
    return {"passed": passed, "recomputed": actual, "recorded": claimed}


def compare_artifacts(native: dict, reference: dict, *, logprob_atol: float) -> dict:
    if native["status"] != "completed" or reference["status"] != "completed":
        raise ValueError("Both inference runs must complete")
    if required(native, "cases_sha256", "native") != required(
        reference, "cases_sha256", "reference"
    ):
        raise ValueError("Runs used different case files")
    if not math.isfinite(logprob_atol) or logprob_atol < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    pairing = pairing_evidence(native, reference)
    rows = validation_rows(native, native=True)
    ref_native = reference.get("kind") == "native"
    references = validation_rows(reference, native=ref_native)
    isolation = isolation_evidence(native, rows)
    ref_isolation = isolation_evidence(reference, references) if ref_native else None
    used = set()
    results = []
    for row in rows:
        candidates = [
            ref
            for ref in references
            if ref["case_id"] == row["case_id"]
            and (
                (ref_native and ref["run_id"] == row["run_id"])
                or (
                    not ref_native
                    and (
                        ref.get("trajectory") == "greedy"
                        or ref["run_id"] == row["run_id"]
                    )
                )
            )
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one matching reference for {row['run_id']}")
        ref = dict(candidates[0])
        used.add(ref["run_id"])
        if ref_native:
            ref.update(trajectory="greedy", greedy_ids=ref["output_ids"])
        if ref.get("trajectory") not in ("greedy", "native_teacher_forced"):
            raise ValueError("Unknown reference trajectory")
        if len(ref["greedy_ids"]) != len(ref["output_ids"]):
            raise ValueError("Incomplete reference greedy IDs")
        if ref["trajectory"] == "greedy" and ref["greedy_ids"] != ref["output_ids"]:
            raise ValueError("Greedy reference output differs from its greedy IDs")
        if ref["max_new_tokens"] != row["max_new_tokens"]:
            raise ValueError("Runs used different output limits")
        margins = ref.get("top2_logit_margins")
        if margins is not None and (
            len(margins) != len(ref["output_ids"])
            or any(not math.isfinite(float(value)) or value < 0 for value in margins)
        ):
            raise ValueError("Invalid reference top-2 margins")
        result = compare_sequences(row, ref, logprob_atol=logprob_atol)
        result["greedy_disagreements"] = [
            {
                "step": step,
                "native_token_id": left,
                "reference_greedy_token_id": right,
                "reference_top2_logit_margin": (
                    margins[step] if margins is not None else None
                ),
                "conditioning_history_matches": ref["trajectory"]
                == "native_teacher_forced"
                or step <= result["common_output_prefix_tokens"],
            }
            for step, (left, right) in enumerate(
                zip(row["output_ids"], ref["greedy_ids"])
            )
            if left != right
        ]
        results.append(result)
    if used != {row["run_id"] for row in references}:
        raise ValueError("Reference contains unmatched validation requests")
    isolated = isolation["passed"] and (
        ref_isolation is None or ref_isolation["passed"]
    )
    return {
        "status": "completed",
        "pairing": pairing,
        "strict_pass": isolated and all(row["strict_pass"] for row in results),
        "request_isolation": isolation,
        "reference_request_isolation": ref_isolation,
        "results": results,
        "all_steps_within_logprob_tolerance": isolated
        and all(row["all_native_steps_within_logprob_tolerance"] for row in results),
        "teacher_forced_all_steps_within_logprob_tolerance": (
            (
                isolated
                and all(
                    row["all_native_steps_within_logprob_tolerance"] for row in results
                )
            )
            if all(
                row["reference_trajectory"] == "native_teacher_forced"
                for row in results
            )
            else None
        ),
        "logprob_atol": logprob_atol,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--logprob-atol", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_artifacts(
        json.loads(args.native.read_text()),
        json.loads(args.reference.read_text()),
        logprob_atol=args.logprob_atol,
    )
    result["native_artifact"] = str(args.native)
    result["reference_artifact"] = str(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("strict_pass", "all_steps_within_logprob_tolerance")
            }
        )
    )


if __name__ == "__main__":
    main()
