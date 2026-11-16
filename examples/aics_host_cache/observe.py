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

"""Observe native cache operations and release only L1 between simulated turns."""

import json
import os
import time
from pathlib import Path


def worker(args):
    import torch

    from examples.aics_qwen35_tp2.hooks import configure_precision
    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.engine.event_loop import EventLoop
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    rank = args.mapping.rank
    torch.npu.set_device(rank)
    precision = configure_precision()
    folder = Path(os.environ["AICS_CACHE_RUN"])
    path = folder / f"rank-{rank}.jsonl"

    def emit(kind, **fields):
        with path.open("a") as out:
            out.write(json.dumps(dict(kind=kind, t=time.monotonic(), **fields)) + "\n")

    emit(
        "worker",
        rank=rank,
        physical=os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")[rank],
        precision=precision,
    )
    pending = []
    sequence = 0
    last_poll = 0.0

    def counts(loop):
        s = loop.scheduler
        return dict(
            device_total=loop._scheduler_cache_geometry.num_usable_pages,
            device_empty=s.empty_lcm_blocks(),
            device_available=s.available_lcm_blocks(),
            device_active=s.active_lcm_blocks(),
            host_pinned=s.host_pool_pinned_blocks(),
            host_cached_entries=s.host_pool_cached_blocks(),
            host_free_parents=s.host_pool_free_blocks(),
            host_evicted_entries=s.host_evicted_blocks(),
            allocated=torch.npu.memory_allocated(rank),
            reserved=torch.npu.memory_reserved(rank),
        )

    process = EventLoop._process_new_requests

    def poll(loop):
        nonlocal last_poll, sequence
        now = time.monotonic()
        if now - last_poll > 0.008:
            last_poll = now
            signal = folder / f"reclaim-{sequence}"
            if signal.is_file() and loop.scheduler.can_clear_cache():
                before = counts(loop)
                # Only warm-up clears both tiers; subsequent boundaries keep Host entries.
                clear = (
                    loop.scheduler.clear_cache
                    if sequence == 0
                    else loop.scheduler.clear_l1_cache
                )
                if clear():
                    while pending:
                        kind, first, last_event, detail = pending.pop(0)
                        last_event.synchronize()
                        emit(
                            kind,
                            turn=sequence - 1,
                            stream_ms=first.elapsed_time(last_event),
                            **detail,
                        )
                    after = counts(loop)
                    emit("reclaim", turn=sequence - 1, before=before, after=after)
                    (folder / f"ack-{sequence}-{rank}").touch()
                    sequence += 1
        return process(loop)

    EventLoop._process_new_requests = poll
    orig = HostCacheExecutor.__init__

    def init(ex, *a, **kw):
        orig(ex, *a, **kw)
        store = ex.host_storage
        emit(
            "layout",
            host_allocated=store.host_buffer_bytes,
            host_parents=store.num_host_lcm_blocks,
            device_parents=ex.layout.num_lcm_blocks,
            groups=[
                dict(
                    id=g.group_id,
                    packing=g.cache_blocks_per_lcm_block,
                    bytes=store.host_cache_block_bytes[i],
                    fields=[f.field_id for f in g.fields],
                )
                for i, g in enumerate(ex.layout.groups)
            ],
        )

    HostCacheExecutor.__init__ = init

    def wrap(method, name, stream_name):
        def submit(ex, op_ids, transfers, *a, **kw):
            if not transfers:
                return method(ex, op_ids, transfers, *a, **kw)
            stream = getattr(ex, stream_name)
            start = torch.npu.Event(enable_timing=True)
            stop = torch.npu.Event(enable_timing=True)
            start.record(stream)
            t = time.monotonic()
            result = method(ex, op_ids, transfers, *a, **kw)
            stop.record(stream)
            pending.append(
                (
                    name,
                    start,
                    stop,
                    dict(
                        rows=[list(v) for v in transfers],
                        ops=list(op_ids),
                        bytes=sum(
                            ex.host_storage.host_cache_block_bytes[g]
                            for g, d, h in transfers
                        ),
                        submit_ms=(time.monotonic() - t) * 1e3,
                    ),
                )
            )
            return result

        return submit

    for n, m, s in [
        ("_start_writing", "d2h", "write_stream"),
        ("_start_loading", "h2d", "load_stream"),
    ]:
        setattr(HostCacheExecutor, n, wrap(getattr(HostCacheExecutor, n), m, s))
    execute = ModelExecutor.execute_forward_op

    def forward(ex, op, *a, **kw):
        first = torch.npu.Event(enable_timing=True)
        last = torch.npu.Event(enable_timing=True)
        first.record()
        r = execute(ex, op, *a, **kw)
        last.record()
        tables = {str(k): v.tolist() for k, v in op.block_tables_arrays().items()}
        pending.append(
            (
                "forward",
                first,
                last,
                dict(
                    prefix=list(op.extend_prefix_lens),
                    inputs=list(op.input_lengths),
                    extends=op.num_extends(),
                    tables=tables,
                ),
            )
        )
        return r

    ModelExecutor.execute_forward_op = forward


def install():
    from multiprocessing.process import BaseProcess

    old = BaseProcess.run

    def run(proc):
        target = proc._target
        if (
            target
            and target.__module__ == "tokenspeed.runtime.engine.event_loop"
            and target.__name__ == "run_event_loop"
        ):
            worker(proc._args[0])
        return old(proc)

    BaseProcess.run = run
