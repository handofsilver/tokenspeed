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

"""Opt-in, process-local NPU allocator sampling for spawned Engine workers."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from precision import read_npu_precision


def start_worker_sampler() -> None:
    """Start only in a spawn child; never create an NPU context for sampling."""
    directory = os.environ.get("AICS_WORKER_MEMORY_DIR")
    if not directory:
        return
    import torch
    import torch_npu  # noqa: F401

    interval = float(os.environ["AICS_WORKER_MEMORY_INTERVAL_S"])

    def sample() -> None:
        path = Path(directory) / f"worker-{os.getpid()}.jsonl"
        with path.open("a", buffering=1) as output:
            while True:
                try:
                    if torch.npu.is_initialized():
                        # Exactly one chip is exposed; avoid querying the parent's
                        # allocator or accidentally selecting another physical chip.
                        row = {
                            "pid": os.getpid(),
                            "monotonic_s": time.perf_counter(),
                            "npu_precision": read_npu_precision(),
                            "allocated_bytes": torch.npu.memory_allocated(0),
                            "reserved_bytes": torch.npu.memory_reserved(0),
                            "lifetime_peak_allocated_bytes": torch.npu.max_memory_allocated(
                                0
                            ),
                            "lifetime_peak_reserved_bytes": torch.npu.max_memory_reserved(
                                0
                            ),
                        }
                        try:
                            free_bytes, total_bytes = torch.npu.mem_get_info(0)
                            row.update(
                                device_driver_free_bytes=free_bytes,
                                device_driver_total_bytes=total_bytes,
                                device_driver_used_bytes=total_bytes - free_bytes,
                            )
                        except Exception as exc:
                            # Keep allocator evidence if the optional driver
                            # query is unavailable on a particular CANN build.
                            row["driver_query_error"] = repr(exc)
                        output.write(json.dumps(row) + "\n")
                except Exception as exc:
                    output.write(
                        json.dumps({"pid": os.getpid(), "error": repr(exc)}) + "\n"
                    )
                    return
                time.sleep(interval)

    threading.Thread(target=sample, name="aics-worker-memory", daemon=True).start()


def summarize_memory(directory: Path, *, start_s: float, end_s: float) -> dict:
    """Summarize only samples in this host-monotonic interval, grouped by PID."""
    workers = []
    errors = []
    for path in sorted(directory.glob("worker-*.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A concurrent worker may not have finished its final line yet.
                continue
            if "error" in row:
                errors.append(row)
            elif start_s <= row["monotonic_s"] <= end_s:
                rows.append(row)
        if rows:
            keys = (
                "allocated_bytes",
                "reserved_bytes",
                "lifetime_peak_allocated_bytes",
                "lifetime_peak_reserved_bytes",
            )
            workers.append(
                {
                    "pid": rows[0]["pid"],
                    "sample_count": len(rows),
                    "first_sample": rows[0],
                    "last_sample": rows[-1],
                    "maxima": {key: max(row[key] for row in rows) for key in keys},
                    "device_driver": {
                        "scope": "whole visible chip, including driver/context and any other processes",
                        "sample_count": sum(
                            "device_driver_used_bytes" in row for row in rows
                        ),
                        "sampled_max_used_bytes": max(
                            (
                                row["device_driver_used_bytes"]
                                for row in rows
                                if "device_driver_used_bytes" in row
                            ),
                            default=None,
                        ),
                        "query_errors": sorted(
                            {
                                row["driver_query_error"]
                                for row in rows
                                if "driver_query_error" in row
                            }
                        ),
                    },
                }
            )
    return {
        "scope": "worker-local torch.npu allocator, not whole-chip driver HBM",
        "interval_sampled_only": True,
        "workers": workers,
        "errors": errors,
        "no_samples": not workers,
    }
