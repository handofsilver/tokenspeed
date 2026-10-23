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

"""Interval-based analysis of per-rank Ascend Chrome traces.

Only device task metadata contributes to device time. CPU launch ranges are
used for flow attribution and are reported separately from device work.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

KINDS = frozenset(
    {
        "compute",
        "communication",
        "communication_component",
        "transfer",
        "wait",
        "unknown",
    }
)
CPU_CATEGORIES = frozenset(
    {"cpu_op", "python_function", "user_annotation", "enqueue", "dequeue"}
)
COMPUTE_TASK_TYPES = frozenset(
    {
        "KERNEL_AICORE",
        "KERNEL_AIVEC",
        "KERNEL_AICPU",
        "KERNEL_MIX_AIC",
        "KERNEL_MIX_AIV",
        "AI_CORE",
        "AI_CPU",
        "AI_VECTOR",
        "AI_VECTOR_CORE",
        "MIX_AIC",
        "MIX_AIV",
    }
)
STAGE_NAMES = frozenset({"aics.prefill", "aics.decode"})
# CANN's KfcOP records identify these AiCPU kernels as HCCL components.
# Their durations must not be counted as model computation or added to the
# enclosing collective's duration.
AICPU_COMMUNICATION_KERNELS = frozenset(
    {"allreduceAicpuKernel", "allgatherAicpuKernel", "broadcastAicpuKernel"}
)


def _collective_operation(event: dict, communication_pids: set[str]) -> str | None:
    """Recognize CANN's device collective envelope, not an enqueue/API range."""
    if (
        event.get("cat") in CPU_CATEGORIES
        or str(event.get("pid")) not in communication_pids
    ):
        return None
    args = event.get("args", {})
    if not {"connection_id", "data_type", "alg_type", "count"} <= args.keys():
        return None
    match = re.match(r"^hcom_([A-Za-z]+)__", str(event.get("name", "")))
    return re.sub(r"(?<!^)(?=[A-Z])", "_", match.group(1)).lower() if match else None


def _hardware_task_key(event: dict) -> tuple[str, str, str]:
    args = event.get("args", {})
    return tuple(
        str(args.get(name)) for name in ("Physic Stream Id", "Task Id", "Model Id")
    )


def _communication_task_key(event: dict) -> tuple[str, str, str]:
    args = event.get("args", {})
    return tuple(str(args.get(name)) for name in ("stream id", "task id", "model id"))


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return disjoint half-open intervals; zero-width events contribute no time."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Invalid interval: {(start, end)}")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    duration = 0.0
    while i < len(left) and j < len(right):
        duration += max(
            0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0])
        )
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return duration


def _decimal(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Trace timestamps must be finite")
    return result


def _event_key(event: dict) -> tuple[str, str, Decimal]:
    return str(event.get("pid")), str(event.get("tid")), _decimal(event["ts"])


def _task_type(event: dict) -> str:
    return str(event.get("args", {}).get("Task Type", "")).upper()


def _engine_kind(task_type: str) -> str:
    if task_type in {"KERNEL_AICPU", "AI_CPU"}:
        return "aicpu"
    if task_type.startswith(("SDMA", "RDMA", "DMA", "PCIE_DMA", "MEMCPY")):
        return "transfer"
    if task_type in COMPUTE_TASK_TYPES:
        return "aicore_or_aivec"
    return "other"


def _field(event: dict, name: str) -> Any:
    return (
        event.get("args", {}).get(name[5:])
        if name.startswith("args.")
        else event.get(name)
    )


def _validate_rules(rules: list[dict]) -> None:
    for rule in rules:
        if rule.get("kind") not in KINDS:
            raise ValueError(f"Unsupported task-rule kind: {rule.get('kind')}")
        if not isinstance(rule.get("evidence"), str) or not rule["evidence"].strip():
            raise ValueError("Every task rule needs a nonempty evidence description")
        match = rule.get("match")
        if not isinstance(match, dict) or not match:
            raise ValueError("Every task rule needs explicit matching fields")
        for name, expected in match.items():
            field = name.removesuffix("_regex")
            if field not in {"name", "cat", "pid", "tid"} and not field.startswith(
                "args."
            ):
                raise ValueError(f"Unsupported task-rule field: {name}")
            if name.endswith("_regex"):
                re.compile(expected)


def _matches(event: dict, match: dict) -> bool:
    for name, expected in match.items():
        value = _field(event, name.removesuffix("_regex"))
        if name.endswith("_regex"):
            if value is None or re.search(expected, str(value)) is None:
                return False
        elif isinstance(expected, list):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def _classify(
    event: dict,
    rules: list[dict],
    *,
    collective_operation: str | None,
    communication_component: bool,
) -> tuple[str, str, int | None]:
    for index, rule in enumerate(rules):
        if _matches(event, rule["match"]):
            return rule["kind"], rule["evidence"], index
    task = _task_type(event)
    args = event.get("args", {})
    if collective_operation is not None:
        return (
            "communication",
            "CANN Communication timeline with collective/connection/data/algorithm metadata",
            None,
        )
    if communication_component or (
        task in {"AI_CPU", "KERNEL_AICPU"}
        and event.get("name") in AICPU_COMMUNICATION_KERNELS
    ):
        return (
            "communication_component",
            "HCCL task identity or AiCPU communication kernel verified by CANN KfcOP",
            None,
        )
    if (
        task == "HCCL"
        or task.startswith("HCCL_")
        or str(args.get("Communication Type", "")).upper() == "HCCL"
    ):
        return "communication", "Explicit HCCL device-task metadata", None
    if re.search(r"hccl|hcom", str(event.get("name", "")), re.IGNORECASE):
        return (
            "unknown",
            "HCCL-like name without verified communication task metadata",
            None,
        )
    if task in COMPUTE_TASK_TYPES:
        return "compute", f"CANN Task Type={task}", None
    if _engine_kind(task) == "transfer":
        return (
            "transfer",
            f"CANN Task Type={task}; transport alone does not prove HCCL",
            None,
        )
    if task.startswith(("EVENT_RECORD", "EVENT_WAIT", "NOTIFY_RECORD", "NOTIFY_WAIT")):
        return "wait", f"CANN Task Type={task}", None
    return (
        "unknown",
        "Device timeline event without a recognized task classification",
        None,
    )


def _device_event(event: dict, hardware_pids: set[str]) -> bool:
    if event.get("cat") in CPU_CATEGORIES:
        return False
    return bool(_task_type(event)) or str(event.get("pid")) in hardware_pids


def _flow_id(event: dict) -> str:
    return json.dumps(
        [event.get("name"), event.get("id", event.get("id2")), event.get("scope")],
        sort_keys=True,
    )


def _stage_at(event: dict, stages: list[dict]) -> str:
    timestamp = _decimal(event["ts"])
    candidates = [
        stage
        for stage in stages
        if str(event.get("pid")) == str(stage.get("pid"))
        and str(event.get("tid")) == str(stage.get("tid"))
        and _decimal(stage["ts"])
        <= timestamp
        < _decimal(stage["ts"]) + _decimal(stage["dur"])
    ]
    if not candidates:
        return "unattributed"
    # Nested duplicate profiler annotations must not count a kernel twice.
    chosen = min(candidates, key=lambda stage: _decimal(stage["dur"]))
    return chosen["name"].removeprefix("aics.")


def _clip(
    interval: tuple[float, float], window: tuple[float, float]
) -> tuple[float, float]:
    start = max(window[0], min(interval[0], window[1]))
    end = min(interval[1], window[1])
    return start, max(start, end)


def _metrics(rows: list[dict], window: tuple[float, float]) -> dict:
    by_kind = {
        kind: [_clip(row["interval"], window) for row in rows if row["kind"] == kind]
        for kind in KINDS
    }
    compute = interval_duration(by_kind["compute"])
    communication = by_kind["communication"] + by_kind["communication_component"]
    comm = interval_duration(communication)
    overlap = intersection_duration(by_kind["compute"], communication)
    activity = interval_duration([_clip(row["interval"], window) for row in rows])
    span = window[1] - window[0]
    return {
        "window_start_relative_us": window[0],
        "window_end_relative_us": window[1],
        "window_duration_us": span,
        "compute_union_us": compute,
        "communication_union_us": comm,
        "collective_envelope_union_us": interval_duration(by_kind["communication"]),
        "compute_communication_overlap_us": overlap,
        "compute_without_communication_us": max(0.0, compute - overlap),
        "exposed_communication_us": max(0.0, comm - overlap),
        "transfer_union_us": interval_duration(by_kind["transfer"]),
        "communication_component_union_us": interval_duration(
            by_kind["communication_component"]
        ),
        "communication_wait_union_us": interval_duration(
            [
                _clip(row["interval"], window)
                for row in rows
                if row["kind"] == "communication_component"
                and _task_type(row["event"]).startswith(("NOTIFY_WAIT", "EVENT_WAIT"))
            ]
        ),
        "wait_union_us": interval_duration(by_kind["wait"]),
        "unknown_union_us": interval_duration(by_kind["unknown"]),
        "aicpu_union_us": interval_duration(
            [_clip(row["interval"], window) for row in rows if row["engine"] == "aicpu"]
        ),
        "aicpu_compute_union_us": interval_duration(
            [
                _clip(row["interval"], window)
                for row in rows
                if row["engine"] == "aicpu" and row["kind"] == "compute"
            ]
        ),
        "aicpu_communication_component_union_us": interval_duration(
            [
                _clip(row["interval"], window)
                for row in rows
                if row["engine"] == "aicpu" and row["kind"] == "communication_component"
            ]
        ),
        "sdma_or_dma_union_us": interval_duration(
            [
                _clip(row["interval"], window)
                for row in rows
                if row["engine"] == "transfer"
            ]
        ),
        "device_activity_union_us": activity,
        "idle_us": max(0.0, span - activity),
        "exposed_communication_fraction": (comm - overlap) / span if span else None,
        "event_counts": dict(Counter(row["kind"] for row in rows)),
    }


def _groups(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        event = row["event"]
        key = (
            row.get("collective_operation") or event.get("name", ""),
            _task_type(event),
            event.get("cat", ""),
            row["kind"],
            row["basis"],
        )
        group = groups.setdefault(
            key,
            {
                "name": key[0],
                "task_type": key[1],
                "category": key[2],
                "kind": key[3],
                "classification_evidence": key[4],
                "count": 0,
                "duration_sum_us": 0.0,
                "intervals": [],
            },
        )
        group["count"] += 1
        group["duration_sum_us"] += row["interval"][1] - row["interval"][0]
        group["intervals"].append(row["interval"])
    output = []
    for group in groups.values():
        group["duration_union_us"] = interval_duration(group.pop("intervals"))
        output.append(group)
    return sorted(
        output, key=lambda group: (-group["duration_union_us"], group["name"])
    )


def summarize_trace(
    trace: list | dict,
    *,
    rank: int,
    window_us: tuple | None = None,
    task_rules: list[dict] | None = None,
    top_n: int = 20,
) -> dict:
    """Analyze one rank, retaining unrecognized device tasks for trace calibration.

    ``window_us`` uses the trace's absolute microsecond timestamps. When omitted,
    the window spans the first to last captured device task, including unknown
    tasks. Rules match device-event fields only and need an evidence description;
    they cannot promote a CPU HCCL API range to device communication.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rules = [] if task_rules is None else task_rules
    _validate_rules(rules)
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("A Chrome trace must contain an event list")
    complete, invalid = [], 0
    for event in events:
        if event.get("ph") != "X":
            continue
        try:
            if _decimal(event["dur"]) < 0:
                raise ValueError("negative duration")
            _decimal(event["ts"])
        except (KeyError, ValueError, InvalidOperation):
            invalid += 1
            continue
        complete.append(event)
    hardware_pids = {
        str(event.get("pid"))
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and str(event.get("args", {}).get("name", "")).startswith("Ascend Hardware")
    }
    communication_pids = {
        str(event.get("pid"))
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Communication"
    }
    collectives = [
        event
        for event in complete
        if _collective_operation(event, communication_pids) is not None
    ]
    communication_connections = {
        str(event["args"]["connection_id"]) for event in collectives
    }
    communication_details = [
        event
        for event in complete
        if str(event.get("pid")) in communication_pids
        and {"stream id", "task id", "model id", "task type"}
        <= event.get("args", {}).keys()
    ]
    communication_task_keys = {
        _communication_task_key(event) for event in communication_details
    }
    device = [
        event
        for event in complete
        if _device_event(event, hardware_pids)
        or _collective_operation(event, communication_pids) is not None
    ]
    origin = min((_decimal(event["ts"]) for event in device), default=Decimal(0))
    stages = [
        event
        for event in complete
        if event.get("name") in STAGE_NAMES and not _device_event(event, hardware_pids)
    ]
    starts = defaultdict(list)
    for event in events:
        if event.get("name") == "torch_to_npu" and event.get("ph") == "s":
            starts[_flow_id(event)].append(event)
    linked, flow_finishes, ambiguous = {}, 0, 0
    for event in events:
        if event.get("name") != "torch_to_npu" or event.get("ph") != "f":
            continue
        flow_finishes += 1
        sources = starts.get(_flow_id(event), [])
        if len(sources) != 1:
            ambiguous += len(sources) > 1
            continue
        try:
            linked[_event_key(event)] = _stage_at(sources[0], stages)
        except (KeyError, ValueError, InvalidOperation):
            continue
    rows, rule_counts = [], Counter()
    for event in device:
        collective_operation = _collective_operation(event, communication_pids)
        communication_component = (
            str(event.get("args", {}).get("connection_id")) in communication_connections
            or _hardware_task_key(event) in communication_task_keys
        )
        kind, basis, rule_index = _classify(
            event,
            rules,
            collective_operation=collective_operation,
            communication_component=communication_component,
        )
        start = float(_decimal(event["ts"]) - origin)
        # Subtract the large epoch timestamp in Decimal before converting to
        # float; CANN commonly stores sub-microsecond ts values as strings.
        interval = (start, start + float(_decimal(event["dur"])))
        rows.append(
            {
                "event": event,
                "interval": interval,
                "kind": kind,
                "basis": basis,
                "engine": _engine_kind(_task_type(event)),
                "collective_operation": collective_operation,
                "stage": linked.get(_event_key(event), "unattributed"),
                "linked": _event_key(event) in linked,
            }
        )
        if rule_index is not None:
            rule_counts[rule_index] += 1
    if window_us is not None:
        if len(window_us) != 2:
            raise ValueError("window_us needs start and end timestamps")
        begin, end = (_decimal(value) for value in window_us)
        if end < begin:
            raise ValueError("window end precedes its start")
        window = (float(begin - origin), float(end - origin))
        window_source = "explicit trace timestamps"
    else:
        window = (
            min((row["interval"][0] for row in rows), default=0.0),
            max((row["interval"][1] for row in rows), default=0.0),
        )
        window_source = "first through last captured device task"
    rows = [
        {**row, "interval": _clip(row["interval"], window)}
        for row in rows
        if (row["interval"][0] < window[1] and row["interval"][1] > window[0])
        or (
            row["interval"][0] == row["interval"][1]
            and window[0] <= row["interval"][0] <= window[1]
        )
    ]
    stage_output = {}
    for stage in sorted(
        {"prefill", "decode", "unattributed"} | {row["stage"] for row in rows}
    ):
        selected = [row for row in rows if row["stage"] == stage]
        stage_window = (
            min((row["interval"][0] for row in selected), default=window[0]),
            max((row["interval"][1] for row in selected), default=window[0]),
        )
        stage_window = _clip(stage_window, window)
        stage_output[stage] = _metrics(selected, stage_window)
        stage_output[stage]["collective_operation_counts"] = dict(
            Counter(
                row["collective_operation"]
                for row in selected
                if row["collective_operation"] is not None
            )
        )
        all_compute = [
            _clip(row["interval"], stage_window)
            for row in rows
            if row["kind"] == "compute"
        ]
        own_communication = [
            _clip(row["interval"], stage_window)
            for row in selected
            if row["kind"] in {"communication", "communication_component"}
        ]
        cross_stage_overlap = intersection_duration(own_communication, all_compute)
        stage_output[stage][
            "communication_overlap_all_compute_us"
        ] = cross_stage_overlap
        stage_output[stage]["exposed_communication_against_all_compute_us"] = max(
            0.0, interval_duration(own_communication) - cross_stage_overlap
        )
        stage_output[stage]["scope"] = (
            "Stage task unions; use rank metrics for idle and cross-stage overlap. "
            "exposed_communication_against_all_compute_us also accounts for "
            "compute attributed to other stages."
        )
    cpu_comm = [
        event
        for event in complete
        if str(event.get("name", "")).startswith("aics.comm.")
        and not _device_event(event, hardware_pids)
    ]
    return {
        "rank": rank,
        "trace_event_count": len(events),
        "invalid_complete_event_count": invalid,
        "timestamp_origin_us": str(origin),
        "window_source": window_source,
        "device_evidence_present": any(
            row["interval"][1] > row["interval"][0] for row in rows
        ),
        "communication_evidence_present": any(
            row["kind"] == "communication" and row["interval"][1] > row["interval"][0]
            for row in rows
        ),
        "stage_evidence_present": all(
            stage_output[stage]["compute_union_us"] > 0
            for stage in ("prefill", "decode")
        ),
        "metrics": _metrics(rows, window),
        "stages": stage_output,
        "cpu_stage_range_counts": dict(
            Counter(event["name"].removeprefix("aics.") for event in stages)
        ),
        "cpu_collective_api": {
            "count": len(cpu_comm),
            "inclusive_duration_sum_us": sum(float(event["dur"]) for event in cpu_comm),
            "interpretation": "Host launch/API ranges; excluded from all device timing metrics",
        },
        "flow_attribution": {
            "finish_count": flow_finishes,
            "ambiguous_source_count": ambiguous,
            "linked_device_event_count": sum(row["linked"] for row in rows),
            "linked_collective_envelope_count": sum(
                row["linked"] and row["collective_operation"] is not None
                for row in rows
            ),
            "unattributed_compute_event_count": sum(
                row["kind"] == "compute" and row["stage"] == "unattributed"
                for row in rows
            ),
            "unattributed_compute_union_us": interval_duration(
                [
                    row["interval"]
                    for row in rows
                    if row["kind"] == "compute" and row["stage"] == "unattributed"
                ]
            ),
            "unattributed_device_event_count": sum(
                row["stage"] == "unattributed" for row in rows
            ),
        },
        "device_task_type_counts": dict(
            Counter(_task_type(event) or "<missing>" for event in device)
        ),
        "device_pid_event_counts": dict(
            Counter(str(row["event"].get("pid")) for row in rows)
        ),
        "communication_timeline_pids": sorted(communication_pids),
        "collective_operation_counts": dict(
            Counter(
                row["collective_operation"]
                for row in rows
                if row["collective_operation"] is not None
            )
        ),
        "communication_detail_task_type_counts": dict(
            Counter(event["args"]["task type"] for event in communication_details)
        ),
        "communication_component_task_counts": dict(
            Counter(
                str(row["event"].get("name"))
                for row in rows
                if row["kind"] == "communication_component"
            )
        ),
        "top_compute_tasks": _groups([row for row in rows if row["kind"] == "compute"])[
            :top_n
        ],
        "top_communication_tasks": _groups(
            [row for row in rows if row["kind"] == "communication"]
        )[:top_n],
        "unknown_tasks": _groups([row for row in rows if row["kind"] == "unknown"]),
        "classification_rules": [
            {**rule, "matched_event_count": rule_counts[index]}
            for index, rule in enumerate(rules)
        ],
        "limitations": [
            "Intervals are unioned per rank across streams; event duration sums are diagnostic only.",
            "Communication time unions CANN collective envelopes and identified physical HCCL components; collective_envelope_union_us retains the envelope-only measurement.",
            "Exposed communication is communication not overlapping captured compute, not a proven critical-path penalty.",
            "Idle means a gap without a captured device task inside the selected trace window; it is not a hardware utilization measurement.",
            "CANN collective envelopes include device synchronization and peer waiting; internal component/wait unions overlap those envelopes and must not be added to them.",
            "Communication detail lanes may contain estimated durations; only matched physical hardware task intervals enter component/wait metrics.",
            "AiCPU HCCL control kernels are excluded from model compute; DMA alone does not imply HCCL.",
            "Stages use torch_to_npu flow sources, never CPU/device timestamp containment.",
            "Profiler synchronization and capture overhead can alter overlap; use separate unprofiled runs for benchmark latency.",
        ],
    }


def summarize_rank_traces(
    traces: dict[int, list | dict],
    *,
    task_rules: list[dict] | None = None,
    top_n: int = 20,
) -> dict:
    """Summarize ranks independently without summing them into wall latency."""
    if not traces:
        raise ValueError("At least one rank trace is required")
    ranks = [
        summarize_trace(trace, rank=rank, task_rules=task_rules, top_n=top_n)
        for rank, trace in sorted(traces.items())
    ]
    return {
        "schema_version": 1,
        "ranks": ranks,
        "rank_count": len(ranks),
        "all_ranks_have_device_evidence": all(
            rank["device_evidence_present"] for rank in ranks
        ),
        "all_ranks_have_communication_evidence": all(
            rank["communication_evidence_present"] for rank in ranks
        ),
        "max_rank_device_window_us": max(
            rank["metrics"]["window_duration_us"] for rank in ranks
        ),
        "max_rank_exposed_communication_us": max(
            rank["metrics"]["exposed_communication_us"] for rank in ranks
        ),
        "interpretation": "Per-rank device windows are diagnostic. Their sum or maximum is not request end-to-end latency; use separately measured host request timings. Cross-rank timestamps are not assumed aligned.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, metavar="RANK=PATH")
    parser.add_argument("--task-rules", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = {}
    for item in args.trace:
        rank, path = item.split("=", 1)
        rank = int(rank)
        if rank in traces:
            parser.error(f"Duplicate trace for rank {rank}")
        traces[rank] = json.loads(Path(path).read_text())
    rules = json.loads(args.task_rules.read_text()) if args.task_rules else None
    result = summarize_rank_traces(traces, task_rules=rules, top_n=args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
