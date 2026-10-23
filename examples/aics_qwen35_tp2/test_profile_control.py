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

"""Profile transitions converge even when one unavailable device operation fails."""

from __future__ import annotations

import concurrent.futures
import threading
import unittest

from examples.aics_qwen35_tp2.profile_native import run_profile_transition


class ReplicaDecision:
    """CPU peer rendezvous in place of an unavailable two-rank process group."""

    def __init__(self):
        self.local = [None, None]
        self.barrier = threading.Barrier(2)

    def agree(self, rank, success):
        self.local[rank] = success
        self.barrier.wait(timeout=2)
        return all(self.local)


class DeviceProfiler:
    """Only the device profiler is unavailable in these pure-data tests."""

    def __init__(self, fail_start):
        self.fail_start = fail_start
        self.active = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("device profiler startup failed")
        self.active = True

    def stop(self):
        self.active = False


class ProfileControlTests(unittest.TestCase):
    def run_ranks(self, failed_rank):
        decision = ReplicaDecision()
        profilers = [
            DeviceProfiler(fail_start=rank == failed_rank) for rank in range(2)
        ]

        def run(rank):
            return run_profile_transition(
                operation=profilers[rank].start,
                agree=lambda ok: decision.agree(rank, ok),
                rollback=profilers[rank].stop,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, rank) for rank in range(2)]
            results = [future.result(timeout=3) for future in futures]
        return profilers, results

    def test_both_devices_start_before_success(self):
        profilers, results = self.run_ranks(failed_rank=None)
        self.assertTrue(all(result["success"] for result in results))
        self.assertTrue(all(profiler.active for profiler in profilers))

    def test_peer_startup_failure_rolls_back_healthy_device_without_hanging(self):
        profilers, results = self.run_ranks(failed_rank=1)
        self.assertFalse(any(result["success"] for result in results))
        self.assertFalse(any(profiler.active for profiler in profilers))
        self.assertIn("Another TP rank", results[0]["message"])
        self.assertIn("device profiler startup failed", results[1]["message"])

    def test_rank_zero_failure_also_reaches_the_replica_decision(self):
        profilers, results = self.run_ranks(failed_rank=0)
        self.assertFalse(any(result["success"] for result in results))
        self.assertFalse(any(profiler.active for profiler in profilers))

    def test_cleanup_failure_is_reported(self):
        def cleanup():
            raise RuntimeError("device cleanup failed")

        result = run_profile_transition(
            operation=lambda: None, agree=lambda ok: False, rollback=cleanup
        )
        self.assertFalse(result["success"])
        self.assertIn("device cleanup failed", result["message"])


if __name__ == "__main__":
    unittest.main()
