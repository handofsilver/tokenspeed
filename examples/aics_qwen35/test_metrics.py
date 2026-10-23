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

import json
import tempfile
import unittest
from pathlib import Path

from metrics import StreamRecord, compare_sequences
from worker_memory import summarize_memory


def frame(ids, count, logprobs=None):
    meta = {"completion_tokens": count, "prompt_tokens": 17, "cached_tokens": 0}
    if logprobs is not None:
        meta["output_token_logprobs"] = [
            (lp, tid, None) for tid, lp in zip(ids, logprobs)
        ]
    return {"output_ids": ids, "meta_info": meta, "text": ""}


def sequence(ids, logprobs, *, forced=False, greedy=None):
    return {
        "case_id": "A",
        "run_id": "A-before",
        "input_ids": [1, 2],
        "output_ids": ids,
        "token_logprobs": logprobs,
        "trajectory": "native_teacher_forced" if forced else "greedy",
        "greedy_ids": ids if greedy is None else greedy,
    }


class TestStreaming(unittest.TestCase):
    def test_repeated_tokens_and_coalesced_later_frame(self):
        record = StreamRecord(start_s=10.0)
        record.add(frame([7], 1, [-0.1]), arrival_s=10.5)
        record.add(frame([7, 8], 3, [-0.2, -0.3]), arrival_s=11.5)
        result = record.finish(end_s=11.6)
        self.assertEqual(result["output_ids"], [7, 7, 8])
        self.assertEqual(result["token_logprobs"], [-0.1, -0.2, -0.3])
        self.assertAlmostEqual(result["observed_ttft_s"], 0.5)
        self.assertAlmostEqual(result["observed_tpot_s"], 0.5)
        self.assertAlmostEqual(result["request_duration_s"], 1.6)
        self.assertFalse(result["all_deliveries_single_token"])

    def test_first_delivery_coalesced_does_not_invent_tpot(self):
        record = StreamRecord(start_s=0.0)
        record.add(frame([7, 7], 2), arrival_s=0.5)
        record.add(frame([8], 3), arrival_s=1.0)
        result = record.finish(end_s=1.1)
        self.assertIsNone(result["observed_tpot_s"])
        self.assertEqual(result["first_delivery_token_count"], 2)

    def test_one_token_and_empty_final_frame(self):
        record = StreamRecord(start_s=0.0)
        record.add(frame([7], 1), arrival_s=0.5)
        record.add(frame([], 1), arrival_s=0.7)
        result = record.finish(end_s=0.8)
        self.assertEqual(result["actual_output_tokens"], 1)
        self.assertIsNone(result["observed_tpot_s"])
        self.assertEqual(result["observed_decode_span_s"], 0.0)

    def test_cumulative_stream_is_rejected(self):
        record = StreamRecord(start_s=0.0)
        record.add(frame([7], 1), arrival_s=0.5)
        with self.assertRaisesRegex(ValueError, "Stream count mismatch"):
            record.add(frame([7, 8], 2), arrival_s=1.0)

    def test_misaligned_logprobs_are_rejected(self):
        record = StreamRecord(start_s=0.0)
        data = frame([7], 1, [-0.1])
        data["meta_info"]["output_token_logprobs"][0] = (-0.1, 8, None)
        with self.assertRaisesRegex(ValueError, "logprob token IDs"):
            record.add(data, arrival_s=0.5)


class TestNumericalComparison(unittest.TestCase):
    def test_equal_trajectory_tolerance(self):
        result = compare_sequences(
            sequence([7, 8], [-0.1, -0.2]),
            sequence([7, 8], [-0.11, -0.21]),
            logprob_atol=0.02,
        )
        self.assertTrue(result["strict_pass"])
        self.assertEqual(result["comparable_logprob_count"], 2)

    def test_diverged_histories_are_not_compared_numerically(self):
        result = compare_sequences(
            sequence([7, 8, 9], [-0.1, -5.0, -6.0]),
            sequence([7, 5, 9], [-0.1, -0.2, -0.3]),
            logprob_atol=0.02,
        )
        self.assertFalse(result["strict_pass"])
        self.assertEqual(result["first_output_mismatch_step"], 1)
        self.assertEqual(result["absolute_logprob_errors"], [0.0])

    def test_teacher_forcing_compares_every_step_but_keeps_greedy_failure(self):
        result = compare_sequences(
            sequence([7, 8, 9], [-0.1, -0.2, -0.3]),
            sequence([7, 8, 9], [-0.1, -0.2, -0.3], forced=True, greedy=[7, 5, 9]),
            logprob_atol=0.02,
        )
        self.assertFalse(result["strict_pass"])
        self.assertEqual(result["comparable_logprob_count"], 3)
        self.assertTrue(result["all_native_steps_within_logprob_tolerance"])

    def test_incomplete_logprobs_do_not_pass(self):
        result = compare_sequences(
            sequence([7, 8], [-0.1]), sequence([7, 8], [-0.1, -0.2]), logprob_atol=0.02
        )
        self.assertFalse(result["strict_pass"])
        self.assertFalse(result["logprobs_complete_and_finite"])

    def test_nonfinite_logprobs_do_not_pass(self):
        result = compare_sequences(
            sequence([7], [float("inf")]), sequence([7], [-0.1]), logprob_atol=0.02
        )
        self.assertFalse(result["strict_pass"])

    def test_wrong_teacher_trajectory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different trajectory"):
            compare_sequences(
                sequence([7, 8], [-0.1, -0.2]),
                sequence([7, 9], [-0.1, -0.2], forced=True),
                logprob_atol=0.02,
            )


class TestMemoryReduction(unittest.TestCase):
    def test_interval_and_process_scope_with_partial_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = [
                {
                    "pid": 100,
                    "monotonic_s": timestamp,
                    "allocated_bytes": value,
                    "reserved_bytes": value * 2,
                    "lifetime_peak_allocated_bytes": 1000,
                    "lifetime_peak_reserved_bytes": 2000,
                }
                for timestamp, value in ((1.0, 900), (2.0, 200), (3.0, 300))
            ]
            (path / "worker-100.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + '\n{"incomplete":'
            )
            result = summarize_memory(path, start_s=1.5, end_s=2.5)
            self.assertEqual(result["workers"][0]["sample_count"], 1)
            self.assertEqual(result["workers"][0]["maxima"]["allocated_bytes"], 200)
            self.assertEqual(
                result["workers"][0]["maxima"]["lifetime_peak_allocated_bytes"], 1000
            )
            self.assertFalse(result["no_samples"])
            self.assertTrue(
                summarize_memory(path, start_s=4.0, end_s=5.0)["no_samples"]
            )


if __name__ == "__main__":
    unittest.main()
