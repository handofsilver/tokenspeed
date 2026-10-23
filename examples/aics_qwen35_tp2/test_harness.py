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

"""Pure-data tests; no model or device initialization is needed."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from examples.aics_qwen35_tp2.memory import summarize_memory
from examples.aics_qwen35_tp2.metrics import (
    StreamRecord,
    benchmark_summary,
    concurrent_group,
    validate_worker_evidence,
    visible_chips,
    worker_device_plan,
)
from examples.aics_qwen35_tp2.run import (
    native_engine_kwargs,
    parser,
    run_group,
    selected_cases,
    sha256,
)

TEST_TMP = Path(__file__).resolve().parents[3] / "tmp" / "hw2-harness-tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class ScriptedEngine:
    """Stand in for unavailable NPU execution while exercising stream isolation."""

    def __init__(self, concurrency):
        self.barrier = threading.Barrier(concurrency)

    def generate(self, *, input_ids, sampling_params, return_logprob, stream):
        self.barrier.wait(timeout=5)
        for index in range(sampling_params["max_new_tokens"]):
            yield {
                "output_ids": [input_ids[0] + index],
                "text": "",
                "meta_info": {
                    "completion_tokens": index + 1,
                    "prompt_tokens": len(input_ids),
                    "cached_tokens": 0,
                    "output_token_logprobs": (
                        [[-0.1, input_ids[0] + index]] if return_logprob else []
                    ),
                },
            }


class HarnessTests(unittest.TestCase):
    def native_args(self):
        return parser().parse_args(
            [
                "native",
                "--model",
                "model",
                "--cases",
                "cases.json",
                "--suite",
                "benchmark",
                "--tp-size",
                "2",
                "--concurrency",
                "4",
                "--attention-backend",
                "mha",
                "--sampling-backend",
                "greedy",
                "--overlap-schedule",
                "off",
                "--record-collectives",
                "off",
                "--prefill-chunk-tokens",
                "512",
                "--memory-utilization",
                "0.2",
                "--output",
                "output.json",
            ]
        )

    def worker(self, rank):
        return {
            "rank": rank,
            "pid": 1000 + rank,
            "physical_chip": rank + 2,
            "logical_device": rank,
            "actual_device": rank,
            "precision": {"matches_requested": True},
            "precision_after_forward": {"matches_requested": True},
            "world_size": 2,
            "attention_tp_size": 2,
            "dense_tp_size": 2,
            "forward_device_ids": [rank],
            "memory_device_ids": [rank],
            "communication_backend": "HcclBackend",
            "successful_forward_count": 4,
            "successful_prefill_count": 1,
            "successful_decode_count": 3,
            "memory_sample_count": 10,
            "sampled_max_allocated_bytes": 100,
            "forward_errors": [],
        }

    def test_physical_visibility_mapping(self):
        chips = visible_chips("2,3", expected=2)
        first = worker_device_plan(rank=0, nprocs=2, base_device=0, chips=chips)
        second = worker_device_plan(rank=1, nprocs=2, base_device=0, chips=chips)
        self.assertEqual((first["logical_device"], first["physical_chip"]), (0, 2))
        self.assertEqual((second["logical_device"], second["physical_chip"]), (1, 3))
        self.assertEqual(visible_chips("7", expected=1), [7])

    def test_invalid_visibility_rejected(self):
        for value, expected in (
            ("", 1),
            ("2,2", 2),
            ("2,3,4", 2),
            ("2, 3", 2),
            ("-1", 1),
            ("2,3", 1),
        ):
            with self.subTest(value=value, expected=expected):
                with self.assertRaises(ValueError):
                    visible_chips(value, expected=expected)
        with self.assertRaises(ValueError):
            worker_device_plan(rank=1, nprocs=2, base_device=1, chips=[2, 3])

    def test_engine_configuration_uses_tensor_parallelism(self):
        args = self.native_args()
        options = native_engine_kwargs(args, output_logprobs=True)
        for name in (
            "world_size",
            "nprocs_per_node",
            "attn_tp_size",
            "dense_tp_size",
            "lm_head_tp_size",
        ):
            self.assertEqual(options[name], 2)
        self.assertEqual(options["data_parallel_size"], 1)
        self.assertEqual(options["max_num_seqs"], 4)
        self.assertEqual(options["max_total_tokens"], 16384)
        self.assertEqual(options["gpu_memory_utilization"], 0.2)
        self.assertTrue(options["disable_overlap_schedule"])
        args.overlap_schedule = "on"
        self.assertFalse(
            native_engine_kwargs(args, output_logprobs=False)[
                "disable_overlap_schedule"
            ]
        )

    def test_concurrent_group_throughput_uses_batch_makespan(self):
        rows = [
            {
                "request_id": "a",
                "actual_output_tokens": 8,
                "requested_output_count_met": True,
            },
            {
                "request_id": "b",
                "actual_output_tokens": 8,
                "requested_output_count_met": True,
            },
        ]
        group = concurrent_group(rows, start_s=10, end_s=14)
        self.assertEqual(group["output_tokens_per_s"], 4)
        self.assertEqual(group["requests_per_s"], 0.5)
        rows[1]["requested_output_count_met"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            concurrent_group(rows, start_s=10, end_s=14)

    def test_streaming_requests_remain_separate(self):
        case = {
            "case_id": "test",
            "input_ids": [7, 8],
            "input_tokens": 2,
            "max_new_tokens": 4,
        }
        rows, group = run_group(
            ScriptedEngine(4),
            case,
            concurrency=4,
            phase="measurement",
            group_id="group",
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(set(row["request_id"] for row in rows)), 4)
        self.assertTrue(all(row["output_ids"] == [7, 8, 9, 10] for row in rows))
        self.assertEqual(group["actual_output_tokens"], 16)
        self.assertTrue(
            all(
                row["request_start_monotonic_s"] >= group["start_monotonic_s"]
                for row in rows
            )
        )

    def test_coalesced_delivery_does_not_invent_tpot(self):
        stream = StreamRecord(start_s=0)
        stream.add(
            {"output_ids": [1, 2], "meta_info": {"completion_tokens": 2}}, arrival_s=1
        )
        self.assertIsNone(stream.finish(end_s=2)["observed_tpot_s"])

    def test_benchmark_summary_excludes_warmup(self):
        rows = [
            {
                "case_id": "x",
                "phase": "measurement",
                "observed_ttft_s": 1,
                "observed_tpot_s": 2,
                "request_duration_s": 3,
            },
            {
                "case_id": "x",
                "phase": "warmup",
                "observed_ttft_s": 100,
                "observed_tpot_s": 100,
                "request_duration_s": 100,
            },
        ]
        groups = [
            {
                "case_id": "x",
                "phase": "measurement",
                "output_tokens_per_s": 4,
                "requests_per_s": 2,
                "makespan_s": 3,
            }
        ]
        result = benchmark_summary(rows, groups)[0]
        self.assertEqual(result["observed_ttft_s"]["median"], 1)
        self.assertEqual(result["output_tokens_per_s"]["median"], 4)

    def test_worker_proof_requires_both_ranks_and_inference(self):
        workers = [self.worker(0), self.worker(1)]
        self.assertTrue(validate_worker_evidence(workers, tp_size=2)["passed"])
        self.assertFalse(validate_worker_evidence(workers[:1], tp_size=2)["passed"])
        workers[1]["successful_decode_count"] = 0
        self.assertFalse(validate_worker_evidence(workers, tp_size=2)["passed"])

    def test_wrong_device_memory_and_precision_fail(self):
        for key, value in (
            ("memory_device_ids", [0]),
            ("forward_device_ids", [0]),
            ("actual_device", 0),
            ("precision_after_forward", {"matches_requested": False}),
            ("communication_backend", "other"),
        ):
            with self.subTest(key=key):
                workers = [self.worker(0), self.worker(1)]
                workers[1][key] = value
                self.assertFalse(validate_worker_evidence(workers, tp_size=2)["passed"])

    def test_memory_reduction_keeps_devices_and_intervals_separate(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            directory = Path(temporary)
            for rank, allocated in ((0, 10), (1, 20)):
                rows = [
                    {
                        "rank": rank,
                        "pid": 10 + rank,
                        "logical_device": rank,
                        "monotonic_s": timestamp,
                        "allocated_bytes": value,
                        "reserved_bytes": value + 1,
                        "lifetime_peak_allocated_bytes": 99,
                        "lifetime_peak_reserved_bytes": 100,
                    }
                    for timestamp, value in ((1, allocated), (5, allocated + 100))
                ]
                (directory / f"rank-{rank}-memory.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n"
                )
            result = summarize_memory(directory, start_s=0, end_s=2)
            self.assertEqual(
                [worker["maxima"]["allocated_bytes"] for worker in result["workers"]],
                [10, 20],
            )
            self.assertEqual(
                [worker["logical_device"] for worker in result["workers"]], [0, 1]
            )

    def test_case_identity_checks_content_without_binding_absolute_path(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}")
            (directory / "tokenizer.json").write_text("{}")
            cases_path = directory / "cases.json"
            data = {
                "model": "old/location",
                "model_config_sha256": sha256(directory / "config.json"),
                "tokenizer_sha256": sha256(directory / "tokenizer.json"),
                "cases": [
                    {
                        "case_id": "A17",
                        "suite": "validation",
                        "input_ids": [1],
                        "input_tokens": 1,
                        "max_new_tokens": 4,
                    }
                ],
            }
            cases_path.write_text(json.dumps(data))
            args = self.native_args()
            args.model, args.cases, args.suite = (
                str(directory),
                str(cases_path),
                "smoke",
            )
            self.assertEqual(selected_cases(args)[0]["case_id"], "A17")
            (directory / "tokenizer.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                selected_cases(args)


if __name__ == "__main__":
    unittest.main()
