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

"""Compare an Ascend Engine with and without Host prefix storage."""

import argparse
import json
import os
import time
from argparse import Namespace
from pathlib import Path

from examples.aics_host_cache.observe import install
from examples.aics_qwen35_tp2.run import native_engine_kwargs, run_request

if __name__ == "__mp_main__":
    import torch_npu

    install()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--offload", choices=["on", "off"], required=True)
    p.add_argument("--budget-ratio", type=float, required=True)
    p.add_argument("--pattern", choices=["high", "low"], required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--num-requests", type=int, default=8)
    p.add_argument("--length", type=int, default=257)
    p.add_argument("--output-tokens", type=int, default=4)
    p.add_argument("--device-tokens", type=int, default=16384)
    a = p.parse_args()
    if (
        a.tp not in (1, 2)
        or a.output_tokens < 2
        or a.num_requests < 2
        or a.budget_ratio <= 0
        or a.length < 2
        or a.length + ((a.num_requests - 1) // 2 + 1) * (a.output_tokens + 1) > 1024
    ):
        p.error("invalid parallelism, lengths, turn count or Host budget")
    folder = a.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    os.environ["AICS_CACHE_RUN"] = str(folder)
    import torch_npu

    from examples.aics_qwen35_tp2.hooks import configure_precision
    from tokenspeed.runtime.entrypoints.engine import Engine

    configure_precision()
    config = native_engine_kwargs(
        Namespace(
            model=a.model,
            tp_size=a.tp,
            concurrency=1,
            max_model_len=1024,
            max_total_tokens=a.device_tokens,
            prefill_chunk_tokens=512,
            memory_utilization=0.50 if "9B" in a.model else 0.15,
            attention_backend="mha",
            sampling_backend="greedy",
            overlap_schedule="off",
            distributed_timeout_seconds=120,
        ),
        output_logprobs=True,
    )
    config.update(
        enable_prefix_caching=True,
        disable_kvstore=a.offload == "off",
        kvstore_ratio=a.budget_ratio,
        kvstore_io_backend="direct",
    )
    rows = []
    out = folder / "requests.json"
    settings = vars(a) | {"run_dir": str(folder)}

    def save(status):
        out.write_text(
            json.dumps(dict(config=settings, status=status, requests=rows), indent=2)
            + "\n"
        )

    save("running")
    engine = Engine(**config)

    def reclaim(i):
        (folder / f"reclaim-{i}").touch()
        end = time.monotonic() + 40
        while not all((folder / f"ack-{i}-{r}").exists() for r in range(a.tp)):
            if time.monotonic() > end:
                raise TimeoutError(f"L1 release {i} did not finish")
            time.sleep(0.01)

    try:
        # Warm up with a disjoint prompt and reset both tiers before measurement.
        warm = [928, 179, 182] * 85 + [34, 56]
        run_request(
            engine,
            dict(
                input_ids=warm, input_tokens=len(warm), max_new_tokens=a.output_tokens
            ),
            request_id="warm",
            phase="validation",
            run_id="warm",
            group_id="warm",
            barrier=None,
        )
        reclaim(0)
        histories = {}
        for i in range(a.num_requests):
            session = i % 2 if a.pattern != "low" else i
            # Prefixes diverge at their very first token; repeated turns extend the actual generated history.
            ids = histories.get(
                session,
                [1010 + session]
                + [
                    2000 + (k * 17 + session * 31) % 1200
                    for k in range(a.length - 1 + (i // 2) * (a.output_tokens + 1))
                ],
            )
            row = run_request(
                engine,
                dict(
                    input_ids=ids, input_tokens=len(ids), max_new_tokens=a.output_tokens
                ),
                request_id=str(i),
                phase="validation",
                run_id=str(i),
                group_id=str(session),
                barrier=None,
            )
            histories[session] = ids + list(row["output_ids"]) + [13]
            reclaim(i + 1)
            row["turn_cycle_s"] = time.perf_counter() - row["request_start_monotonic_s"]
            rows.append(row)
            save("running")
        save("complete")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
