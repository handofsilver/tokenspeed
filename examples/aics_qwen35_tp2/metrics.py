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

"""Pure-data reduction for TP validation and concurrent streaming."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from examples.aics_qwen35.metrics import StreamRecord, compare_sequences


def visible_chips(value: str, *, expected: int) -> list[int]:
    parts = value.split(",")
    if expected not in (1, 2) or len(parts) != expected:
        raise ValueError("Expose exactly one chip for TP1 or two chips for TP2")
    if any(not part.isdecimal() for part in parts):
        raise ValueError(
            "Visible chip IDs must be comma-separated nonnegative integers"
        )
    chips = [int(part) for part in parts]
    if len(set(chips)) != len(chips):
        raise ValueError("Visible chip IDs must not repeat")
    return chips


def worker_device_plan(
    *, rank: int, nprocs: int, base_device: int, chips: list[int]
) -> dict:
    if nprocs != len(chips) or not 0 <= rank < nprocs:
        raise ValueError(
            "This harness supports one host with one worker per visible chip"
        )
    logical = rank % nprocs + base_device
    if not 0 <= logical < len(chips):
        raise ValueError("Worker logical device is outside the visible chip list")
    return {
        "rank": rank,
        "local_rank": rank,
        "logical_device": logical,
        "physical_chip": chips[logical],
        "visible_physical_chips": chips,
    }


def statistics_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Metric contains a nonfinite value")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def concurrent_group(rows: list[dict], *, start_s: float, end_s: float) -> dict:
    if not rows or end_s <= start_s:
        raise ValueError("Concurrent group must have requests and a positive duration")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Concurrent request IDs must be unique")
    if any(not row["requested_output_count_met"] for row in rows):
        raise ValueError("Cannot summarize a group with incomplete output")
    total = sum(row["actual_output_tokens"] for row in rows)
    duration = end_s - start_s
    return {
        "concurrency": len(rows),
        "start_monotonic_s": start_s,
        "end_monotonic_s": end_s,
        "makespan_s": duration,
        "actual_output_tokens": total,
        "output_tokens_per_s": total / duration,
        "requests_per_s": len(rows) / duration,
        "request_ids": [row["request_id"] for row in rows],
    }


def benchmark_summary(rows: list[dict], groups: list[dict]) -> list[dict]:
    summaries = []
    for case_id in sorted({row["case_id"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["case_id"] == case_id and row["phase"] == "measurement"
        ]
        selected_groups = [
            group
            for group in groups
            if group["case_id"] == case_id and group["phase"] == "measurement"
        ]
        item = {
            "case_id": case_id,
            "requests": len(selected),
            "groups": len(selected_groups),
        }
        for key in ("observed_ttft_s", "observed_tpot_s", "request_duration_s"):
            item[key] = statistics_summary(
                [row[key] for row in selected if row[key] is not None]
            )
        for key in ("output_tokens_per_s", "requests_per_s", "makespan_s"):
            item[key] = statistics_summary([group[key] for group in selected_groups])
        summaries.append(item)
    return summaries


def validate_worker_evidence(workers: list[dict], *, tp_size: int) -> dict:
    ranks = [worker["rank"] for worker in workers]
    reasons = []
    if sorted(ranks) != list(range(tp_size)):
        reasons.append(f"Expected ranks 0..{tp_size - 1}; found {ranks}")
    if len({worker["pid"] for worker in workers}) != tp_size:
        reasons.append("Workers do not have distinct PIDs")
    if len({worker["physical_chip"] for worker in workers}) != tp_size:
        reasons.append("Workers do not have distinct physical chips")
    for worker in workers:
        if not worker["precision"]["matches_requested"]:
            reasons.append(f"Rank {worker['rank']} precision mismatch")
        if (
            not worker["precision_after_forward"]
            or not worker["precision_after_forward"]["matches_requested"]
        ):
            reasons.append(f"Rank {worker['rank']} precision changed during inference")
        if worker["actual_device"] != worker["logical_device"]:
            reasons.append(f"Rank {worker['rank']} device mismatch")
        if worker["forward_device_ids"] != [worker["logical_device"]] or worker[
            "memory_device_ids"
        ] != [worker["logical_device"]]:
            reasons.append(
                f"Rank {worker['rank']} forward or memory used a different device"
            )
        if tp_size > 1 and worker["communication_backend"] != "HcclBackend":
            reasons.append(f"Rank {worker['rank']} did not use HCCL")
        if (
            worker["world_size"] != tp_size
            or worker["attention_tp_size"] != tp_size
            or worker["dense_tp_size"] != tp_size
        ):
            reasons.append(
                f"Rank {worker['rank']} is not using the requested TP layout"
            )
        if worker["successful_forward_count"] < 2:
            reasons.append(
                f"Rank {worker['rank']} has fewer than two completed forwards"
            )
        if (
            worker["successful_prefill_count"] < 1
            or worker["successful_decode_count"] < 1
        ):
            reasons.append(f"Rank {worker['rank']} lacks prefill or decode execution")
        if (
            worker["memory_sample_count"] < 1
            or worker["sampled_max_allocated_bytes"] <= 0
        ):
            reasons.append(f"Rank {worker['rank']} has no allocator evidence")
        if worker["forward_errors"]:
            reasons.append(f"Rank {worker['rank']} has a failed forward")
    return {"passed": not reasons, "expected_tp_size": tp_size, "reasons": reasons}


def summarize_collectives(events: list[dict]) -> dict:
    complete = [
        event for event in events if event["event"] == "collective" and event["success"]
    ]
    return {
        "count": len(complete),
        "counts_by_operation": dict(Counter(event["operation"] for event in complete)),
        "logical_payload_bytes": sum(event["payload_bytes"] for event in complete),
        "scope": "Host calls and logical input payload; not wire traffic or device communication time",
    }
