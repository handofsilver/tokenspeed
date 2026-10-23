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

import unittest

from profile_native import summarize_trace


def complete(name, start, duration, *, pid=1, tid=1, category="cpu_op", args=None):
    return {
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": tid,
        "cat": category,
        "args": args or {},
    }


class TestProfileSummary(unittest.TestCase):
    def test_async_kernel_uses_flow_source_not_cpu_timestamp_overlap(self):
        trace = [
            complete("aics.prefill", 0, 100),
            complete("aten::linear", 10, 60),
            complete("aten::mm", 20, 30),
            complete("aics.decode", 100, 100),
            complete("aten::mm", 110, 20),
            # The prefill kernel executes during the CPU decode range.
            complete(
                "MatMul",
                140,
                25,
                pid=2,
                tid=4,
                args={"Task Type": "KERNEL_AICORE"},
            ),
            {"ph": "s", "name": "torch_to_npu", "id": 11, "pid": 1, "tid": 1, "ts": 20},
            {
                "ph": "f",
                "name": "torch_to_npu",
                "id": 11,
                "pid": 2,
                "tid": 4,
                "ts": 140,
            },
        ]
        summary = summarize_trace(trace, top_n=20)
        self.assertTrue(summary["cpu_stage_ranges_present"])
        self.assertFalse(summary["stage_evidence_present"])
        self.assertEqual(summary["matrix_operator_count"], 3)
        self.assertEqual(summary["stages"]["prefill"]["kernel_duration_sum_us"], 25)
        self.assertEqual(summary["stages"]["decode"]["kernel_count"], 0)
        rows = {
            row["name"]: row
            for row in summary["stages"]["prefill"]["top_cpu_ops_by_self_thread_time"]
        }
        self.assertEqual(rows["aten::linear"]["inclusive_thread_us"], 60)
        self.assertEqual(rows["aten::linear"]["self_thread_us"], 30)
        self.assertEqual(rows["aten::mm"]["self_thread_us"], 30)

    def test_known_hardware_without_flow_remains_unattributed(self):
        trace = {
            "traceEvents": [
                complete("aics.prefill", 0, 100),
                complete("aics.decode", 100, 100),
                complete("aten::mm", 110, 20),
                complete(
                    "Unlinked",
                    20,
                    10,
                    pid=2,
                    category="hardware",
                    args={"Task Type": "KERNEL_AIVEC"},
                ),
            ]
        }
        summary = summarize_trace(trace, top_n=20)
        self.assertEqual(summary["npu_kernel_event_count"], 1)
        self.assertEqual(summary["linked_kernel_count"], 0)
        self.assertEqual(summary["stages"]["unattributed"]["kernel_count"], 1)
        self.assertTrue(summary["cpu_stage_ranges_present"])
        self.assertEqual(summary["matrix_operator_count"], 1)
        self.assertFalse(summary["stage_evidence_present"])
        self.assertEqual(
            summary["kernel_attribution"]["unattributed_count_fraction"], 1
        )
        self.assertEqual(
            summary["kernel_attribution"]["unattributed_duration_fraction"], 1
        )

    def test_both_stages_require_attributed_kernels_and_report_missing_fraction(self):
        trace = [
            complete("aics.prefill", 0, 100),
            complete("aten::mm", 10, 20),
            complete("aics.decode", 100, 100),
            complete("aten::mm", 110, 20),
            complete(
                "Unlinked",
                230,
                50,
                pid=2,
                category="hardware",
                args={"Task Type": "KERNEL_AICPU"},
            ),
        ]
        for flow_id, source, target, duration in [(1, 10, 220, 30), (2, 110, 260, 20)]:
            trace.extend(
                [
                    complete(
                        "MatMul",
                        target,
                        duration,
                        pid=2,
                        tid=4,
                        args={"Task Type": "KERNEL_AICORE"},
                    ),
                    {
                        "ph": "s",
                        "name": "torch_to_npu",
                        "id": flow_id,
                        "pid": 1,
                        "tid": 1,
                        "ts": source,
                    },
                    {
                        "ph": "f",
                        "name": "torch_to_npu",
                        "id": flow_id,
                        "pid": 2,
                        "tid": 4,
                        "ts": target,
                    },
                ]
            )
        summary = summarize_trace(trace, top_n=20)
        self.assertTrue(summary["stage_evidence_present"])
        self.assertEqual(summary["stages"]["prefill"]["kernel_count"], 1)
        self.assertEqual(summary["stages"]["decode"]["kernel_count"], 1)
        self.assertEqual(
            summary["kernel_attribution"],
            {
                "total_count": 3,
                "unattributed_count": 1,
                "unattributed_count_fraction": 1 / 3,
                "total_duration_sum_us": 100,
                "unattributed_duration_sum_us": 50,
                "unattributed_duration_fraction": 0.5,
            },
        )

    def test_real_cann_task_types_distinguish_unlinked_kernels_and_transfers(self):
        task_types = ["KERNEL_AIVEC", "KERNEL_AICPU", "KERNEL_AICORE", "AI_CORE"]
        trace = [
            complete(
                "device_op", i, 2, pid=2, category=None, args={"Task Type": task_type}
            )
            for i, task_type in enumerate(task_types)
        ]
        trace.extend(
            [
                complete(
                    "MEMCPY_ASYNC",
                    10,
                    3,
                    pid=2,
                    category=None,
                    args={"Task Type": "SDMA_SQE"},
                ),
                complete(
                    "device_unknown",
                    20,
                    5,
                    pid=2,
                    category=None,
                    args={"Task Type": "EVENT_WAIT"},
                ),
            ]
        )
        summary = summarize_trace(trace, top_n=20)
        self.assertEqual(summary["npu_kernel_event_count"], 4)
        self.assertEqual(summary["linked_kernel_count"], 0)
        self.assertEqual(summary["kernel_attribution"]["unattributed_count"], 4)
        self.assertEqual(summary["kernel_attribution"]["total_duration_sum_us"], 8)
        self.assertEqual(summary["npu_transfer_event_count"], 1)
        self.assertEqual(summary["npu_transfer_duration_sum_us"], 3)
        self.assertEqual(summary["stages"]["unattributed"]["transfer_count"], 1)
        self.assertEqual(
            summary["unclassified_device_task_type_counts"], {"EVENT_WAIT": 1}
        )

    def test_linked_copy_does_not_count_as_decode_kernel_evidence(self):
        trace = [
            complete("aics.prefill", 0, 100),
            complete("aten::mm", 10, 20),
            complete("aics.decode", 100, 100),
            complete("aten::mm", 110, 20),
        ]
        device_tasks = [
            (1, 10, 230, 30, "KERNEL_AIVEC"),
            (2, 110, 260, 2, "SDMA_SQE"),
            (3, 115, 270, 9, "UNKNOWN_TASK"),
        ]
        for flow_id, source, target, duration, task_type in device_tasks:
            trace.extend(
                [
                    complete(
                        "device_op",
                        target,
                        duration,
                        pid=2,
                        tid=4,
                        category=None,
                        args={"Task Type": task_type},
                    ),
                    {
                        "ph": "s",
                        "name": "torch_to_npu",
                        "id": flow_id,
                        "pid": 1,
                        "tid": 1,
                        "ts": source,
                    },
                    {
                        "ph": "f",
                        "name": "torch_to_npu",
                        "id": flow_id,
                        "pid": 2,
                        "tid": 4,
                        "ts": target,
                    },
                ]
            )
        summary = summarize_trace(trace, top_n=20)
        self.assertTrue(summary["cpu_stage_ranges_present"])
        self.assertFalse(summary["stage_evidence_present"])
        self.assertEqual(summary["npu_kernel_event_count"], 1)
        self.assertEqual(summary["npu_transfer_event_count"], 1)
        self.assertEqual(summary["linked_transfer_count"], 1)
        self.assertEqual(summary["stages"]["prefill"]["kernel_count"], 1)
        self.assertEqual(summary["stages"]["decode"]["kernel_count"], 0)
        self.assertEqual(summary["stages"]["decode"]["transfer_count"], 1)
        self.assertEqual(summary["stages"]["decode"]["transfer_duration_sum_us"], 2)
        self.assertEqual(
            summary["unclassified_device_task_type_counts"], {"UNKNOWN_TASK": 1}
        )

    def test_unrecognized_events_are_not_invented_as_kernels(self):
        summary = summarize_trace(
            [complete("unknown", 0, 50, category="mystery")], top_n=3
        )
        self.assertEqual(summary["complete_event_categories"], {"mystery": 1})
        self.assertEqual(summary["npu_kernel_event_count"], 0)
        self.assertIsNone(summary["kernel_attribution"]["unattributed_count_fraction"])
        self.assertIsNone(
            summary["kernel_attribution"]["unattributed_duration_fraction"]
        )

    def test_partial_cpu_overlaps_are_reported(self):
        summary = summarize_trace(
            [complete("aten::a", 0, 50), complete("aten::b", 25, 50)], top_n=3
        )
        self.assertEqual(summary["cpu_partial_overlap_count"], 1)


if __name__ == "__main__":
    unittest.main()
