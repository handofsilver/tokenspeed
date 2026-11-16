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

"""Summarize a sequential Host L2 trial against the run without offloading."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def experiment(path):
    run = json.loads((path / "requests.json").read_text())
    require(
        run["status"] == "complete"
        and len(run["requests"]) == run["config"]["num_requests"],
        f"Incomplete trial: {path}",
    )
    return run


def score(row, reference, atol):
    require(
        row["input_ids"] == reference["input_ids"],
        "Cannot compare scores conditioned on different prompt histories",
    )
    a, b = row["output_ids"], reference["output_ids"]
    ap, bp = row["token_logprobs"], reference["token_logprobs"]
    require(
        len(a) == len(b) == len(ap) == len(bp) == row["max_new_tokens"],
        "Incomplete output or logprobs",
    )
    require(all(isfinite(v) for v in ap + bp), "Nonfinite logprob")
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    error = max((abs(x - y) for x, y in zip(ap[:prefix], bp[:prefix])), default=None)
    return dict(
        equal_ids=sum(x == y for x, y in zip(a, b)),
        tokens=len(a),
        max_error=error,
        passed=prefix == len(a) and error is not None and error <= atol,
    )


def rank_summary(path, rank, run):
    data = [
        json.loads(line)
        for line in (path / f"rank-{rank}.jsonl").read_text().splitlines()
    ]
    (worker,) = [e for e in data if e["kind"] == "worker"]
    require(
        worker["rank"] == rank and worker["precision"]["matches_requested"],
        "Invalid worker identity or precision",
    )
    layouts = [e for e in data if e["kind"] == "layout"]
    enabled = run["config"]["offload"] == "on"
    require(len(layouts) == int(enabled), "Wrong number of Host arenas")
    layout = layouts[0] if enabled else None
    (warm,) = [e for e in data if e.get("turn") == -1 and e["kind"] == "reclaim"]
    warm = warm["after"]
    require(
        warm["host_cached_entries"] == 0
        and warm["device_total"] == warm["device_empty"],
        "Warm-up cache not cleared",
    )
    turns = []
    prior = {}
    example = None
    for i, request in enumerate(run["requests"]):
        events = [e for e in data if e.get("turn") == i]
        after = [e for e in events if e["kind"] == "reclaim"]
        require(len(after) == 1, "Missing reclamation")
        b, a = after[0]["before"], after[0]["after"]
        require(
            a["device_empty"] == a["device_total"] == a["device_available"]
            and a["device_active"] == a["host_pinned"] == 0,
            "An idle boundary still owns a device page or Host pin",
        )
        pre = [e for e in events if e["kind"] == "forward" and e["extends"]]
        require(pre, "Missing prefill")
        prefixes = [e["prefix"][0] for e in pre]
        start = prefixes[0]
        n = request["input_tokens"]
        require(
            start == request["cached_tokens"]
            and start + sum(e["inputs"][0] for e in pre) == n,
            "Forward range disagrees with delivered cache hit",
        )
        moves = {k: [e for e in events if e["kind"] == k] for k in ("h2d", "d2h")}
        require(
            bool(moves["h2d"]) == (start > 0),
            "L1 was cleared: reuse must be backed by Host reads",
        )
        pages = {}
        group = request["group_id"]
        for e in pre:
            for g, tables in e["tables"].items():
                for page in tables[0]:
                    if page <= 0:
                        continue
                    key = (g, page)
                    pages[key] = group
                    if example is None and key in prior and prior[key][1] != group:
                        example = dict(
                            group=g, page=page, earlier_turn=prior[key][0], later_turn=i
                        )
        prior.update({k: (i, v) for k, v in pages.items()})
        row = dict(
            turn=i,
            session=group,
            prompt=n,
            restored=start,
            prefill_end=n,
            prefill_ms=sum(e["stream_ms"] for e in pre),
            device_used_before=b["device_total"] - b["device_empty"],
            device_empty_after=a["device_empty"],
            device_available=a["device_available"],
            host_free=a["host_free_parents"],
            host_entries=a["host_cached_entries"],
            host_evictions=a["host_evicted_entries"] - warm["host_evicted_entries"],
            allocated=b["allocated"],
            reserved=b["reserved"],
        )
        for k, ops in moves.items():
            row[k + "_bytes"] = sum(e["bytes"] for e in ops)
            row[k + "_ms"] = sum(e["stream_ms"] for e in ops)
        turns.append(row)
    return dict(
        rank=rank,
        device_total=warm["device_total"],
        host_parents=layout["host_parents"] if layout else 0,
        host_pool_bytes=layout["host_allocated"] if layout else 0,
        groups=(
            [
                dict(id=g["id"], packing=g["packing"], bytes=g["bytes"])
                for g in layout["groups"]
            ]
            if layout
            else []
        ),
        reused_device_page=example,
        turns=turns,
    )


def summarize(path, reference_path, *, atol):
    run = experiment(path)
    reference = experiment(reference_path)
    c = run["config"]
    ref = reference["config"]
    require(ref["offload"] == "off", "Reference must run without Host offloading")
    for key in [
        "tp",
        "length",
        "pattern",
        "output_tokens",
        "device_tokens",
        "num_requests",
    ]:
        require(c[key] == ref[key], f"Comparison changes {key}")
    require(Path(c["model"]).name == Path(ref["model"]).name, "Different models")
    ranks = [rank_summary(path, r, run) for r in range(c["tp"])]
    turns = ranks[0]["turns"]
    positions = [t["restored"] for t in turns]
    for r in ranks:
        require(
            [t["restored"] for t in r["turns"]] == positions,
            "Ranks restored different frontiers",
        )
    requests = run["requests"]
    scores = [score(a, b, atol) for a, b in zip(requests, reference["requests"])]
    total = sum(s["tokens"] for s in scores)
    metrics = dict(
        output_tokens=total,
        mean_cycle_ms=1000 * sum(x["turn_cycle_s"] for x in requests) / len(requests),
        median_ttft_ms=1000 * median(x["observed_ttft_s"] for x in requests),
        cycle_output_tokens_per_s=total / sum(x["turn_cycle_s"] for x in requests),
        hits=sum(t > 0 for t in positions),
        misses=sum(t == 0 for t in positions),
    )
    return dict(
        model=Path(c["model"]).name,
        tp=c["tp"],
        offload=c["offload"],
        ratio=c["budget_ratio"],
        device_tokens=c["device_tokens"],
        pattern=c["pattern"],
        rank_positions_agree=True,
        logprob_atol=atol,
        equal_ids=sum(s["equal_ids"] for s in scores),
        max_logprob_error=max(
            (s["max_error"] for s in scores if s["max_error"] is not None), default=None
        ),
        valid=all(s["passed"] for s in scores),
        metrics=metrics,
        ranks=ranks,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--logprob-atol", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    res = summarize(a.run, a.reference, atol=a.logprob_atol)
    a.output.write_text(json.dumps(res, indent=2) + "\n")
    return 0 if res["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
