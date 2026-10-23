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

"""Per-worker allocator sampling on the worker's assigned logical device."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


def allocator_sample(*, device: int, rank: int, pid: int) -> dict:
    import torch

    row = {
        "rank": rank,
        "pid": pid,
        "logical_device": device,
        "monotonic_s": time.perf_counter(),
        "allocated_bytes": torch.npu.memory_allocated(device),
        "reserved_bytes": torch.npu.memory_reserved(device),
        "lifetime_peak_allocated_bytes": torch.npu.max_memory_allocated(device),
        "lifetime_peak_reserved_bytes": torch.npu.max_memory_reserved(device),
    }
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        row.update(
            device_driver_free_bytes=free_bytes,
            device_driver_total_bytes=total_bytes,
            device_driver_used_bytes=total_bytes - free_bytes,
        )
    except Exception as exc:
        row["driver_query_error"] = repr(exc)
    return row


def start_sampler(
    *, directory: Path, device: int, rank: int, pid: int, interval_s: float
) -> None:
    import torch

    # Capture the assigned device explicitly: a new Python thread otherwise
    # starts with device zero even when rank one runs on device one.
    def sample_loop() -> None:
        torch.npu.set_device(device)
        with (directory / f"rank-{rank}-memory.jsonl").open("a", buffering=1) as output:
            while True:
                try:
                    row = allocator_sample(device=device, rank=rank, pid=pid)
                except Exception as exc:
                    row = {
                        "rank": rank,
                        "pid": pid,
                        "monotonic_s": time.perf_counter(),
                        "error": repr(exc),
                    }
                output.write(json.dumps(row) + "\n")
                if "error" in row:
                    return
                time.sleep(interval_s)

    threading.Thread(
        target=sample_loop, name=f"aics-tp-memory-{rank}", daemon=True
    ).start()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may still be completing the final line.
            continue
    return rows


def summarize_memory(directory: Path, *, start_s: float, end_s: float) -> dict:
    workers = []
    for path in sorted(directory.glob("rank-*-memory.jsonl")):
        rows = [
            row for row in read_jsonl(path) if start_s <= row["monotonic_s"] <= end_s
        ]
        good = [row for row in rows if "error" not in row]
        item = {
            "rank": rows[0]["rank"] if rows else int(path.name.split("-")[1]),
            "sample_count": len(good),
            "errors": [row for row in rows if "error" in row],
        }
        if good:
            keys = (
                "allocated_bytes",
                "reserved_bytes",
                "lifetime_peak_allocated_bytes",
                "lifetime_peak_reserved_bytes",
            )
            item.update(
                logical_device=good[0]["logical_device"],
                pid=good[0]["pid"],
                maxima={key: max(row[key] for row in good) for key in keys},
                sampled_driver_max_used_bytes=max(
                    (
                        row["device_driver_used_bytes"]
                        for row in good
                        if "device_driver_used_bytes" in row
                    ),
                    default=None,
                ),
            )
        workers.append(item)
    return {
        "scope": "per-rank torch.npu allocator; driver usage separately includes other processes",
        "sampled_only": True,
        "workers": workers,
    }
