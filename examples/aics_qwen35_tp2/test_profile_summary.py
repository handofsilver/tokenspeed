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

"""Pure-data regression tests for Ascend device interval accounting."""

import json
import unittest

from examples.aics_qwen35_tp2.profile_summary import (
    intersection_duration,
    merge_intervals,
    summarize_rank_traces,
    summarize_trace,
)


def complete(
    name, start, duration, *, task=None, pid=2, tid=1, category=None, args=None
):
    metadata = dict(args or {})
    if task is not None:
        metadata["Task Type"] = task
    return {
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": tid,
        "cat": category,
        "args": metadata,
    }


def cpu(name, start, duration, *, pid=1, tid=1):
    return complete(name, start, duration, pid=pid, tid=tid, category="cpu_op")


def flow(identifier, source, target, *, target_tid=1, target_pid=2):
    return [
        {
            "ph": "s",
            "name": "torch_to_npu",
            "id": identifier,
            "pid": 1,
            "tid": 1,
            "ts": source,
        },
        {
            "ph": "f",
            "name": "torch_to_npu",
            "id": identifier,
            "pid": target_pid,
            "tid": target_tid,
            "ts": target,
        },
    ]


class TestIntervals(unittest.TestCase):
    def test_duplicate_nested_and_touching_intervals_count_once(self):
        self.assertEqual(
            merge_intervals([(5, 10), (0, 7), (2, 4), (10, 15), (20, 20)]), [(0, 15)]
        )
        self.assertEqual(
            intersection_duration([(0, 10), (5, 15)], [(8, 12), (11, 20)]), 7
        )
        self.assertEqual(intersection_duration([(0, 4)], [(4, 8)]), 0)
        with self.assertRaises(ValueError):
            merge_intervals([(4, 1)])
        with self.assertRaises(ValueError):
            merge_intervals([(0, float("nan"))])

    def test_device_stream_union_and_exposed_communication(self):
        trace = [
            complete("mm", 0, 10, task="KERNEL_AICORE"),
            complete("vec", 5, 10, task="KERNEL_AIVEC", tid=2),
            complete("Cumsum", 14, 4, task="KERNEL_AICPU", tid=3),
            complete("allreduce", 8, 12, task="HCCL", tid=4),
            complete("allgather", 18, 7, task="HCCL_TASK", tid=5),
            complete("MEMCPY_ASYNC", 24, 4, task="SDMA_SQE"),
            complete("new task", 29, 1, task="UNRECOGNIZED_TASK"),
            complete("wait", 31, 2, task="EVENT_WAIT"),
            cpu("aics.comm.all_reduce", 0, 1000),
        ]
        summary = summarize_trace(trace, rank=0, window_us=(-5, 35))
        metrics = summary["metrics"]
        expected = {
            "window_duration_us": 40,
            "compute_union_us": 18,
            "communication_union_us": 17,
            "compute_communication_overlap_us": 10,
            "compute_without_communication_us": 8,
            "exposed_communication_us": 7,
            "transfer_union_us": 4,
            "aicpu_union_us": 4,
            "sdma_or_dma_union_us": 4,
            "unknown_union_us": 1,
            "wait_union_us": 2,
            "device_activity_union_us": 31,
            "idle_us": 9,
        }
        for key, value in expected.items():
            self.assertEqual(metrics[key], value, key)
        self.assertEqual(metrics["exposed_communication_fraction"], 7 / 40)
        self.assertEqual(
            summary["cpu_collective_api"]["inclusive_duration_sum_us"], 1000
        )
        self.assertTrue(summary["communication_evidence_present"])
        self.assertEqual(summary["unknown_tasks"][0]["task_type"], "UNRECOGNIZED_TASK")

    def test_epoch_timestamps_retain_submicrosecond_precision(self):
        summary = summarize_trace(
            [
                complete("a", "1792048736948965.881", "0.125", task="KERNEL_AIVEC"),
                complete("b", "1792048736948966.006", "0.250", task="KERNEL_AIVEC"),
            ],
            rank=0,
        )
        self.assertEqual(summary["metrics"]["compute_union_us"], 0.375)
        self.assertEqual(summary["metrics"]["window_duration_us"], 0.375)

    def test_explicit_window_clips_events_and_excludes_outside_tasks(self):
        summary = summarize_trace(
            [
                complete("mm", -10, 15, task="KERNEL_AICORE"),
                complete("hccl", 5, 15, task="HCCL"),
                complete("outside", 30, 10, task="HCCL"),
            ],
            rank=0,
            window_us=(0, 10),
        )
        self.assertEqual(summary["metrics"]["compute_union_us"], 5)
        self.assertEqual(summary["metrics"]["communication_union_us"], 5)
        self.assertEqual(summary["metrics"]["idle_us"], 0)
        self.assertEqual(summary["metrics"]["event_counts"]["communication"], 1)
        self.assertEqual(len(summary["top_communication_tasks"]), 1)
        self.assertEqual(summary["top_communication_tasks"][0]["duration_sum_us"], 5)


class TestClassification(unittest.TestCase):
    def test_communication_component_tail_counts_once_and_can_overlap_compute(self):
        trace = [
            {
                "ph": "M",
                "name": "process_name",
                "pid": 3,
                "args": {"name": "Communication"},
            },
            complete(
                "hcom_allReduce__909_0_1",
                10,
                10,
                pid=3,
                args={
                    "connection_id": 17,
                    "data_type": "BFP16",
                    "alg_type": "RING",
                    "count": 128,
                },
            ),
            complete("allreduceAicpuKernel", 18, 7, task="AI_CPU"),
            complete("mm", 22, 8, task="AI_CORE"),
        ]
        metrics = summarize_trace(trace, rank=0)["metrics"]
        self.assertEqual(metrics["collective_envelope_union_us"], 10)
        self.assertEqual(metrics["communication_component_union_us"], 7)
        self.assertEqual(metrics["communication_union_us"], 15)
        self.assertEqual(metrics["compute_communication_overlap_us"], 3)
        self.assertEqual(metrics["exposed_communication_us"], 12)

    def test_cann_collective_envelope_excludes_aicpu_control_and_dma_details_from_compute(
        self,
    ):
        trace = [
            {
                "ph": "M",
                "name": "process_name",
                "pid": 3,
                "args": {"name": "Communication"},
            },
            complete("mm", 0, 20, task="AI_VECTOR_CORE"),
            complete(
                "hcom_allReduce__909_0_1",
                10,
                30,
                pid=3,
                args={
                    "connection_id": 17,
                    "data_type": "BFP16",
                    "alg_type": "RING",
                    "count": 128,
                },
            ),
            complete("allreduceAicpuKernel", 12, 3, task="AI_CPU"),
            complete("Cumsum", 42, 3, task="AI_CPU"),
            complete(
                "MEMCPY_ASYNC",
                21,
                2,
                task="SDMA_SQE",
                args={"Physic Stream Id": 5, "Task Id": 7, "Model Id": 4294967295},
            ),
            complete("MEMCPY_ASYNC", 24, 1, task="SDMA_SQE"),
            complete(
                "NOTIFY_WAIT_SQE",
                30,
                5,
                task="NOTIFY_WAIT_SQE",
                args={"Physic Stream Id": 5, "Task Id": 9, "Model Id": 4294967295},
            ),
            complete(
                "Memcpy",
                21,
                2,
                pid=3,
                args={
                    "stream id": 5,
                    "task id": 7,
                    "model id": 4294967295,
                    "task type": "Memcpy",
                },
            ),
            # CANN's detail lane can contain an estimated wait span. Count
            # the matched physical hardware interval above, never this 20us.
            complete(
                "Notify_Wait",
                30,
                20,
                pid=3,
                args={
                    "stream id": 5,
                    "task id": 9,
                    "model id": 4294967295,
                    "task type": "Notify_Wait",
                    "duration estimated(us)": 0.02,
                },
            ),
            cpu("aics.prefill", 0, 50),
            cpu("aics.decode", 100, 50),
        ]
        trace += flow(1, 2, 0) + flow(2, 12, 10, target_pid=3) + flow(3, 110, 42)
        summary = summarize_trace(trace, rank=0)
        metrics = summary["metrics"]
        self.assertTrue(summary["communication_evidence_present"])
        self.assertTrue(summary["stage_evidence_present"])
        self.assertEqual(metrics["compute_union_us"], 23)
        self.assertEqual(metrics["communication_union_us"], 30)
        self.assertEqual(metrics["compute_communication_overlap_us"], 10)
        self.assertEqual(metrics["exposed_communication_us"], 20)
        self.assertEqual(metrics["communication_component_union_us"], 10)
        self.assertEqual(metrics["communication_wait_union_us"], 5)
        self.assertEqual(metrics["aicpu_compute_union_us"], 3)
        self.assertEqual(metrics["aicpu_communication_component_union_us"], 3)
        self.assertEqual(metrics["window_duration_us"], 45)
        self.assertEqual(metrics["idle_us"], 2)
        self.assertEqual(metrics["transfer_union_us"], 1)
        self.assertEqual(summary["collective_operation_counts"], {"all_reduce": 1})
        self.assertEqual(summary["top_communication_tasks"][0]["name"], "all_reduce")
        self.assertEqual(
            summary["stages"]["prefill"]["collective_operation_counts"],
            {"all_reduce": 1},
        )
        self.assertEqual(summary["unknown_tasks"], [])

    def test_cann_collective_needs_process_and_payload_metadata(self):
        base = complete(
            "hcom_allReduce__909_0_1", 0, 10, pid=3, args={"connection_id": 1}
        )
        trace = [
            {
                "ph": "M",
                "name": "process_name",
                "pid": 3,
                "args": {"name": "Communication"},
            },
            base,
        ]
        summary = summarize_trace(trace, rank=0)
        self.assertFalse(summary["communication_evidence_present"])
        base["args"].update(data_type="BFP16", alg_type="RING", count=1)
        base["cat"] = "cpu_op"
        self.assertFalse(
            summarize_trace(trace, rank=0)["communication_evidence_present"]
        )

    def test_cpu_hccl_api_cannot_be_promoted_even_by_an_explicit_rule(self):
        event = cpu("aics.comm.all_reduce", 0, 100)
        event["args"]["Task Type"] = "HCCL"
        rules = [
            {
                "match": {"name_regex": "all_reduce"},
                "kind": "communication",
                "evidence": "Synthetic device-task rule",
            }
        ]
        summary = summarize_trace([event], rank=0, task_rules=rules)
        self.assertFalse(summary["device_evidence_present"])
        self.assertFalse(summary["communication_evidence_present"])
        self.assertEqual(summary["metrics"]["communication_union_us"], 0)
        self.assertEqual(summary["cpu_collective_api"]["count"], 1)
        self.assertEqual(summary["classification_rules"][0]["matched_event_count"], 0)

    def test_dma_and_aicpu_do_not_implicitly_mean_hccl(self):
        trace = [
            complete("memcpy", i * 10, 4, task=task)
            for i, task in enumerate(
                ["SDMA_SQE", "PCIE_DMA_SQE", "RDMA_SQE", "KERNEL_AICPU"]
            )
        ]
        summary = summarize_trace(trace, rank=0)
        self.assertEqual(summary["metrics"]["communication_union_us"], 0)
        self.assertEqual(summary["metrics"]["transfer_union_us"], 12)
        self.assertEqual(summary["metrics"]["compute_union_us"], 4)
        self.assertEqual(summary["metrics"]["aicpu_union_us"], 4)
        self.assertFalse(summary["communication_evidence_present"])

    def test_hccl_like_names_stay_unknown_until_evidence_rule_is_applied(self):
        trace = [
            complete("HcomAllReduce", 0, 8, task="KERNEL_AICPU"),
            complete("Cumsum", 10, 3, task="KERNEL_AICPU"),
        ]
        baseline = summarize_trace(trace, rank=0)
        self.assertEqual(baseline["metrics"]["compute_union_us"], 3)
        self.assertEqual(baseline["metrics"]["communication_union_us"], 0)
        self.assertEqual(baseline["unknown_tasks"][0]["name"], "HcomAllReduce")
        rules = [
            {
                "match": {
                    "args.Task Type": "KERNEL_AICPU",
                    "name_regex": "^HcomAllReduce$",
                },
                "kind": "communication",
                "evidence": "Verified against the synthetic fixture's device task record",
            }
        ]
        summary = summarize_trace(trace, rank=0, task_rules=rules)
        self.assertEqual(summary["metrics"]["communication_union_us"], 8)
        self.assertEqual(summary["metrics"]["aicpu_union_us"], 11)
        self.assertEqual(summary["metrics"]["compute_union_us"], 3)
        self.assertEqual(summary["unknown_tasks"], [])
        self.assertEqual(summary["classification_rules"][0]["matched_event_count"], 1)

    def test_explicit_hccl_metadata_can_identify_a_dma_communication_task(self):
        summary = summarize_trace(
            [
                complete(
                    "copy", 0, 7, task="SDMA_SQE", args={"Communication Type": "HCCL"}
                )
            ],
            rank=0,
        )
        self.assertEqual(summary["metrics"]["communication_union_us"], 7)
        self.assertEqual(summary["metrics"]["transfer_union_us"], 0)
        self.assertEqual(summary["metrics"]["sdma_or_dma_union_us"], 7)

    def test_unknown_hardware_events_without_task_type_are_retained(self):
        summary = summarize_trace(
            [
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": 2,
                    "args": {"name": "Ascend Hardware"},
                },
                complete("future task", 0, 5),
                complete("ordinary CPU range", 0, 100, pid=3),
            ],
            rank=0,
        )
        self.assertEqual(summary["metrics"]["unknown_union_us"], 5)
        self.assertEqual(summary["unknown_tasks"][0]["name"], "future task")
        self.assertEqual(summary["device_task_type_counts"], {"<missing>": 1})

    def test_rules_need_explicit_match_and_evidence(self):
        for rule in [
            {"match": {"name": "x"}, "kind": "communication"},
            {"match": {}, "kind": "communication", "evidence": "fixture"},
            {"match": {"name": "x"}, "kind": "cpu", "evidence": "fixture"},
            {
                "match": {"unrecognized": "x"},
                "kind": "communication",
                "evidence": "fixture",
            },
        ]:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                summarize_trace([], rank=0, task_rules=[rule])


class TestAttribution(unittest.TestCase):
    def test_stage_exposure_accounts_for_compute_from_another_stage(self):
        trace = [
            cpu("aics.prefill", 0, 50),
            cpu("aics.decode", 100, 50),
            complete("prefill compute", 180, 15, task="KERNEL_AICORE"),
            complete("prefill comm", 200, 20, task="HCCL"),
            complete("decode compute", 210, 20, task="KERNEL_AIVEC"),
        ]
        trace += flow(1, 10, 180) + flow(2, 20, 200) + flow(3, 110, 210)
        summary = summarize_trace(trace, rank=0)
        prefill = summary["stages"]["prefill"]
        self.assertEqual(prefill["communication_overlap_all_compute_us"], 10)
        self.assertEqual(prefill["exposed_communication_against_all_compute_us"], 10)
        self.assertEqual(summary["metrics"]["exposed_communication_us"], 10)

    def test_device_stage_comes_from_cpu_flow_source_not_timestamp_containment(self):
        trace = [
            cpu("aics.prefill", 0, 50),
            cpu("aics.prefill", 1, 48),
            cpu("aics.decode", 100, 50),
            complete("prefill device", 160, 10, task="KERNEL_AICORE"),
            complete("decode device", 20, 5, task="KERNEL_AIVEC"),
        ]
        trace += flow(1, 12, 160) + flow(2, 110, 20)
        summary = summarize_trace({"traceEvents": trace}, rank=0)
        self.assertTrue(summary["stage_evidence_present"])
        self.assertEqual(summary["stages"]["prefill"]["compute_union_us"], 10)
        self.assertEqual(summary["stages"]["prefill"]["event_counts"]["compute"], 1)
        self.assertEqual(summary["stages"]["decode"]["compute_union_us"], 5)
        self.assertEqual(summary["flow_attribution"]["linked_device_event_count"], 2)
        self.assertEqual(summary["cpu_stage_range_counts"]["prefill"], 2)

    def test_linked_copy_does_not_prove_decode_compute(self):
        trace = [
            cpu("aics.prefill", 0, 50),
            cpu("aics.decode", 100, 50),
            complete("mm", 160, 10, task="KERNEL_AICORE"),
            complete("copy", 180, 5, task="SDMA_SQE"),
        ]
        trace += flow(1, 12, 160) + flow(2, 110, 180)
        summary = summarize_trace(trace, rank=0)
        self.assertFalse(summary["stage_evidence_present"])
        self.assertEqual(summary["stages"]["decode"]["transfer_union_us"], 5)
        self.assertEqual(summary["stages"]["decode"]["compute_union_us"], 0)

    def test_unlinked_or_ambiguous_flows_do_not_invent_a_stage(self):
        trace = [
            cpu("aics.prefill", 0, 50),
            complete("mm", 5, 10, task="KERNEL_AICORE"),
        ]
        unlinked = summarize_trace(trace, rank=0)
        self.assertEqual(unlinked["stages"]["unattributed"]["compute_union_us"], 10)
        trace += flow(1, 2, 5)
        trace += [
            {"ph": "s", "name": "torch_to_npu", "id": 1, "pid": 1, "tid": 1, "ts": 3}
        ]
        ambiguous = summarize_trace(trace, rank=0)
        self.assertEqual(ambiguous["flow_attribution"]["ambiguous_source_count"], 1)
        self.assertEqual(ambiguous["stages"]["unattributed"]["compute_union_us"], 10)

    def test_two_ranks_remain_independent_and_json_serializable(self):
        summary = summarize_rank_traces(
            {
                0: [complete("mm", 0, 10, task="KERNEL_AICORE")],
                1: [complete("mm", 10000, 20, task="KERNEL_AICORE")],
            }
        )
        self.assertEqual([row["rank"] for row in summary["ranks"]], [0, 1])
        self.assertEqual(summary["ranks"][0]["metrics"]["compute_union_us"], 10)
        self.assertEqual(summary["ranks"][1]["metrics"]["compute_union_us"], 20)
        self.assertEqual(summary["max_rank_device_window_us"], 20)
        self.assertNotIn("end_to_end_us", summary)
        json.dumps(summary, allow_nan=False)

    def test_empty_and_invalid_capture_cannot_pass_evidence_gates(self):
        summary = summarize_trace([complete("invalid", 0, -1, task="HCCL")], rank=0)
        self.assertEqual(summary["invalid_complete_event_count"], 1)
        self.assertFalse(summary["device_evidence_present"])
        self.assertFalse(summary["stage_evidence_present"])
        self.assertIsNone(summary["metrics"]["exposed_communication_fraction"])
        json.dumps(summary, allow_nan=False)
        with self.assertRaises(ValueError):
            summarize_rank_traces({})
        with self.assertRaises(ValueError):
            summarize_trace([], rank=-1)
        with self.assertRaises(ValueError):
            summarize_trace([], rank=0, window_us=(2, 1))


if __name__ == "__main__":
    unittest.main()
