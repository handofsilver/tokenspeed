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

"""Behavioral tests for TP artifact matching."""

from copy import deepcopy

import pytest

from examples.aics_qwen35_tp2.compare import compare_artifacts


def artifact(*, tp_size=2):
    engine = {
        "model": "/models/Qwen3.5-0.8B",
        "device": "npu",
        "base_gpu_id": 0,
        "world_size": tp_size,
        "nprocs_per_node": tp_size,
        "attn_tp_size": tp_size,
        "dense_tp_size": tp_size,
        "lm_head_tp_size": tp_size,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "dtype": "bfloat16",
        "mamba_ssm_dtype": "float32",
        "language_model_only": True,
        "trust_remote_code": False,
        "retraction_snapshot_ratio": 0.0,
        "attention_backend": "mha",
        "sampling_backend": "greedy",
        "logprob_order": "torch",
        "enforce_eager": True,
        "disable_prefill_graph": True,
        "disable_pdl": True,
        "disable_autotune": True,
        "disable_overlap_schedule": True,
        "enable_prefix_caching": False,
        "disable_kvstore": True,
        "enable_output_logprobs": True,
        "max_num_seqs": 1,
        "max_model_len": 1024,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 512,
        "chunked_prefill_size": 512,
        "gpu_memory_utilization": 0.1,
        "stream_interval": 1,
        "stream_output": True,
        "seed": 0,
    }
    rows = [
        {
            "phase": "validation",
            "case_id": case,
            "run_id": run,
            "input_ids": inputs,
            "input_tokens": len(inputs),
            "max_new_tokens": 2,
            "output_ids": [3, 4],
            "actual_output_tokens": 2,
            "token_logprobs": [-0.2, -0.3],
        }
        for case, run, inputs in (
            ("A17", "A-before", [1, 2]),
            ("B129", "B", [1, 5]),
            ("A17", "A-after", [1, 2]),
        )
    ]
    return {
        "status": "completed",
        "kind": "native",
        "cases_sha256": "cases",
        "config": {"tp_size": tp_size},
        "engine_args": engine,
        "model_identity": {
            "model_config_sha256": "config",
            "tokenizer_sha256": "tokenizer",
            "weight_index_sha256": "index",
            "weight_files": [{"name": "model.safetensors", "bytes": 100}],
        },
        "request_isolation": {
            **{
                key: True
                for key in (
                    "same_token_ids",
                    "same_token_logprobs",
                    "logprobs_complete",
                    "intervening_distinct_request",
                )
            },
            "cache_flush_between_requests": False,
        },
        "results": rows,
    }


def hf_reference(native, *, forced):
    rows = deepcopy(native["results"] if forced else native["results"][:2])
    for row in rows:
        row["trajectory"] = "native_teacher_forced" if forced else "greedy"
        row["greedy_ids"] = list(row["output_ids"])
        row["top2_logit_margins"] = [0.1, 0.2]
        if not forced:
            row["run_id"] = row["case_id"]
    return {
        "status": "completed",
        "kind": "reference",
        "cases_sha256": "cases",
        "config": {"dtype": "bfloat16", "cache_mode": "cached"},
        "model_config_sha256": "config",
        "reference_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration",
        "attention_implementation": "eager",
        "results": rows,
    }


def test_tp1_greedy_reference():
    native, reference = artifact(), artifact(tp_size=1)
    for row in reference["results"]:
        row["token_logprobs"][0] += 0.01
    result = compare_artifacts(native, reference, logprob_atol=0.02)
    assert result["strict_pass"]
    assert result["pairing"]["reference_tp_size"] == 1
    assert not compare_artifacts(native, reference, logprob_atol=0.005)["strict_pass"]


def test_hf_greedy_case_can_match_both_occurrences_of_a():
    native = artifact()
    result = compare_artifacts(
        native, hf_reference(native, forced=False), logprob_atol=0.01
    )
    assert result["strict_pass"]
    assert result["pairing"]["model_identity_fields_checked"] == ["model_config_sha256"]
    assert "weight_files" in result["pairing"]["unverified_identity_fields"]
    assert result["teacher_forced_all_steps_within_logprob_tolerance"] is None


def test_teacher_forced_history_stays_separate_from_greedy_agreement():
    native = artifact()
    reference = hf_reference(native, forced=True)
    reference["results"][1]["greedy_ids"] = [3, 5]
    reference["results"][1]["top2_logit_margins"] = [0.1, 0.00001]
    result = compare_artifacts(native, reference, logprob_atol=0.01)
    assert result["all_steps_within_logprob_tolerance"]
    assert result["teacher_forced_all_steps_within_logprob_tolerance"]
    assert not result["strict_pass"]
    assert result["results"][1]["greedy_disagreements"] == [
        {
            "step": 1,
            "native_token_id": 4,
            "reference_greedy_token_id": 5,
            "reference_top2_logit_margin": 0.00001,
            "conditioning_history_matches": True,
        }
    ]


@pytest.mark.parametrize(
    "field",
    ["model_config_sha256", "tokenizer_sha256", "weight_index_sha256", "weight_files"],
)
def test_reject_different_model_identity(field):
    native, reference = artifact(), artifact(tp_size=1)
    reference["model_identity"][field] = (
        [{"name": "other.safetensors", "bytes": 100}]
        if field == "weight_files"
        else "other"
    )
    with pytest.raises(ValueError, match="model identity"):
        compare_artifacts(native, reference, logprob_atol=0.1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("mamba_ssm_dtype", "bfloat16"),
        ("max_num_seqs", 4),
        ("chunked_prefill_size", 128),
        ("disable_overlap_schedule", False),
        ("seed", 9),
        ("some_new_algorithm", "changed"),
    ],
)
def test_reject_changed_engine_settings(field, value):
    native, reference = artifact(), artifact(tp_size=1)
    reference["engine_args"][field] = value
    with pytest.raises(ValueError, match="Engine configuration"):
        compare_artifacts(native, reference, logprob_atol=0.1)


def test_device_mapping_paths_and_memory_budget_may_differ():
    native, reference = artifact(), artifact(tp_size=1)
    reference["engine_args"].update(
        model="/other/path", base_gpu_id=1, gpu_memory_utilization=0.2
    )
    assert compare_artifacts(native, reference, logprob_atol=0.1)["strict_pass"]


@pytest.mark.parametrize(
    "field", ["dtype", "mamba_ssm_dtype", "enable_output_logprobs"]
)
def test_missing_engine_settings_cannot_default_to_pass(field):
    native, reference = artifact(), artifact(tp_size=1)
    del native["engine_args"][field]
    with pytest.raises(ValueError, match="Missing"):
        compare_artifacts(native, reference, logprob_atol=0.1)


def test_inconsistent_parallel_dimensions_rejected():
    native, reference = artifact(), artifact(tp_size=1)
    native["engine_args"]["attn_tp_size"] = 1
    with pytest.raises(ValueError, match="TP dimension"):
        compare_artifacts(native, reference, logprob_atol=0.1)


@pytest.mark.parametrize("field", ["dtype", "cache_mode"])
def test_legacy_reference_requires_actual_declared_precision_and_mode(field):
    native = artifact()
    reference = hf_reference(native, forced=False)
    del reference["config"][field]
    with pytest.raises(ValueError, match="Missing"):
        compare_artifacts(native, reference, logprob_atol=0.1)


def test_hf_model_config_must_match():
    native = artifact()
    reference = hf_reference(native, forced=False)
    reference["model_config_sha256"] = "different"
    with pytest.raises(ValueError, match="model configuration"):
        compare_artifacts(native, reference, logprob_atol=0.1)


def test_hf_weight_precision_must_match():
    native = artifact()
    reference = hf_reference(native, forced=False)
    reference["config"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="weight precision"):
        compare_artifacts(native, reference, logprob_atol=0.1)


@pytest.mark.parametrize("field", ["model_identity", "engine_args"])
def test_missing_native_pairing_evidence_cannot_pass(field):
    native, reference = artifact(), artifact(tp_size=1)
    del native[field]
    with pytest.raises(ValueError, match="Missing"):
        compare_artifacts(native, reference, logprob_atol=0.1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_logprobs_rejected(value):
    native, reference = artifact(), artifact(tp_size=1)
    native["results"][1]["token_logprobs"][0] = value
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_artifacts(native, reference, logprob_atol=0.1)


@pytest.mark.parametrize(
    "change",
    [
        "cases",
        "failed",
        "duplicate",
        "missing",
        "wrong-count",
        "input-count",
        "short-greedy",
        "extra",
    ],
)
def test_reject_unpaired_artifacts(change):
    native, reference = artifact(), artifact(tp_size=1)
    if change == "cases":
        reference["cases_sha256"] = "other"
    elif change == "failed":
        reference["status"] = "failed"
    elif change == "duplicate":
        reference["results"] *= 2
    elif change == "missing":
        reference["results"] = []
    elif change == "wrong-count":
        reference["results"][0]["output_ids"] = [3]
    elif change == "input-count":
        reference["results"][0]["input_tokens"] = 99
    elif change == "short-greedy":
        reference = hf_reference(native, forced=False)
        reference["results"][0]["greedy_ids"] = [3]
    else:
        reference = hf_reference(native, forced=False)
        extra = deepcopy(reference["results"][0])
        extra.update(case_id="extra", run_id="extra")
        reference["results"].append(extra)
    with pytest.raises(ValueError):
        compare_artifacts(native, reference, logprob_atol=0.1)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_tolerance_rejected(value):
    with pytest.raises(ValueError, match="Tolerance"):
        compare_artifacts(artifact(), artifact(tp_size=1), logprob_atol=value)


@pytest.mark.parametrize("side", ["native", "reference"])
def test_incomplete_logprobs_or_state_isolation_cannot_pass(side):
    native, reference = artifact(), artifact(tp_size=1)
    target = native if side == "native" else reference
    target["request_isolation"]["same_token_logprobs"] = False
    assert not compare_artifacts(native, reference, logprob_atol=0.1)["strict_pass"]
    target["request_isolation"]["same_token_logprobs"] = True
    target["results"][0]["token_logprobs"] = [-0.2]
    assert not compare_artifacts(native, reference, logprob_atol=0.1)["strict_pass"]


def test_isolation_is_recomputed_instead_of_trusting_flags():
    native, reference = artifact(), artifact(tp_size=1)
    native["results"][2]["token_logprobs"][0] += 0.001
    assert not compare_artifacts(native, reference, logprob_atol=0.1)["strict_pass"]


def test_unrecorded_cache_flush_policy_cannot_prove_isolation():
    native, reference = artifact(), artifact(tp_size=1)
    del native["request_isolation"]["cache_flush_between_requests"]
    assert not compare_artifacts(native, reference, logprob_atol=0.1)["strict_pass"]
