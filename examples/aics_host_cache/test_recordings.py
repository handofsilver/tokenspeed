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

"""Acceptance checks over completed Engine runs (set AICS_CACHE_RECORDINGS)."""

import json
import os
from pathlib import Path

import pytest

from examples.aics_host_cache.analyze import summarize


@pytest.fixture(scope="module")
def recordings():
    path = os.environ.get("AICS_CACHE_RECORDINGS")
    if not path:
        pytest.skip("set AICS_CACHE_RECORDINGS to the completed run directories")
    return Path(path)


@pytest.mark.parametrize(
    "trial,baseline", [("s2-high-wide", "s2-high-off"), ("m9-high-on", "m9-high-off")]
)
def test_alternating_sessions_restore_the_same_context_on_both_ranks(
    recordings, trial, baseline
):
    result = summarize(recordings / trial, recordings / baseline, atol=0.10)
    assert result["tp"] == 2 and result["valid"] and result["rank_positions_agree"]
    assert result["metrics"]["hits"] == len(result["ranks"][0]["turns"]) - 2
    for rank in result["ranks"]:
        assert all(
            t["restored"] == t["prompt"] - 2 and t["h2d_bytes"] > 0
            for t in rank["turns"][2:]
        )
        trace = (
            (recordings / trial / f"rank-{rank['rank']}.jsonl").read_text().splitlines()
        )
        moves = [
            e for e in map(json.loads, trace) if e["kind"] == "h2d" and e["turn"] >= 2
        ]
        for turn in range(2, len(rank["turns"])):
            # A recurrent checkpoint alone or a KV-only load cannot resume this model.
            assert {
                g for e in moves if e["turn"] == turn for g, _, _ in e["rows"]
            } == set(range(len(rank["groups"])))


def test_eviction_and_miss_recompute(recordings):
    tight = summarize(
        recordings / "s2-high-tight", recordings / "s2-high-off", atol=0.10
    )
    assert tight["valid"] and tight["metrics"]["misses"] > 2
    for rank in tight["ranks"]:
        assert any(
            a["host_evictions"] > 0 and not b["restored"]
            for a, b in zip(rank["turns"][1:], rank["turns"][2:])
        )
        assert all(0 <= t["host_free"] <= rank["host_parents"] for t in rank["turns"])


def test_fresh_sessions_do_not_borrow_another_prefix(recordings):
    result = summarize(recordings / "s2-low-on", recordings / "s2-low-off", atol=0.10)
    assert result["valid"] and result["metrics"]["hits"] == 0
    assert all(t["h2d_bytes"] == 0 for r in result["ranks"] for t in r["turns"])


def test_released_pages_are_reused_by_another_session(recordings):
    result = summarize(
        recordings / "s2-high-wide", recordings / "s2-high-off", atol=0.10
    )
    for rank in result["ranks"]:
        assert rank["reused_device_page"] is not None
        assert all(
            t["device_empty_after"] == rank["device_total"]
            and t["device_used_before"] > 0
            for t in rank["turns"]
        )
