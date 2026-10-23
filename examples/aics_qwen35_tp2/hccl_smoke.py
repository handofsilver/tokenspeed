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

"""Verify two local HCCL ranks before loading model weights."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import time
from pathlib import Path


def worker(rank: int, port: int, output: str, chips: list[int]):
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    torch.npu.set_device(rank)
    torch.npu.set_option(
        {
            "ACL_PRECISION_MODE": "must_keep_origin_dtype",
            "ALLOW_MATMUL_HF32": "disable",
            "ALLOW_CONV_HF32": "disable",
        }
    )
    started = time.time()
    dist.init_process_group(
        "hccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    process_group_manager.register_process_group("hccl", (0, 1), dist.group.WORLD)
    backend = HcclBackend()
    try:
        # Each rank performs a matrix product and contributes a different sum.
        values = torch.full((8,), rank + 1.0, device=f"npu:{rank}")
        matrix = torch.ones((32, 32), device=f"npu:{rank}", dtype=torch.bfloat16)
        product = matrix @ matrix
        backend.all_reduce(values, (0, 1))
        gathered = backend.all_gather(values, (0, 1))
        scattered = backend.reduce_scatter(gathered, (0, 1))
        token_scattered = backend.token_reduce_scatter(
            gathered.reshape(4, 4), (0, 1), [2, 2]
        )
        torch.npu.synchronize()
        reduced = values.cpu().tolist()
        products = product.cpu().tolist()
        all_values = gathered.cpu().tolist()
        scatter_values = scattered.cpu().tolist()
        token_values = token_scattered.cpu().flatten().tolist()
        passed = reduced == [3.0] * 8 and all_values == [3.0] * 16
        passed = passed and scatter_values == [6.0] * 8 and token_values == [6.0] * 8
        passed = passed and all(value == 32.0 for row in products for value in row)
        free, total = torch.npu.mem_get_info(rank)
        result = {
            "status": "ok" if passed else "failed",
            "rank": rank,
            "pid": os.getpid(),
            "logical_device": rank,
            "physical_chip": chips[rank],
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "all_reduce": reduced,
            "all_gather": all_values,
            "reduce_scatter": scatter_values,
            "token_reduce_scatter": token_values,
            "matmul_shape": [32, 32],
            "matmul_value": products[0][0],
            "allocated_bytes": torch.npu.memory_allocated(rank),
            "reserved_bytes": torch.npu.memory_reserved(rank),
            "driver_free_bytes": free,
            "driver_total_bytes": total,
            "elapsed_s": time.time() - started,
        }
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        if not passed:
            raise AssertionError(f"HCCL smoke failed on rank {rank}")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chips = [int(value) for value in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
    if len(chips) != 2 or len(set(chips)) != 2 or min(chips) < 0:
        raise ValueError("Select exactly two distinct chips from one card")
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    import torch.multiprocessing as mp

    mp.spawn(
        worker, args=(port, str(args.output.resolve()), chips), nprocs=2, join=True
    )
    results = [
        json.loads((args.output / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    print(json.dumps({"status": "ok", "ranks": results}, indent=2))


if __name__ == "__main__":
    main()
