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

"""Explicit, verified process-local precision for reference and native runs."""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path

EXPECTED_OPTIONS = {
    "ACL_PRECISION_MODE": "must_keep_origin_dtype",
    "ALLOW_MATMUL_HF32": "disable",
    "ALLOW_CONV_HF32": "disable",
}


def read_npu_precision() -> dict:
    import torch
    import torch_npu

    # torch_npu 2.9.0 exposes no public get_option; its HF32 getters use this
    # same _C accessor. Preserve the actual string, including an unset value.
    options = {}
    for name in EXPECTED_OPTIONS:
        value = torch_npu._C._npu_getOption(name)
        options[name] = value.decode() if isinstance(value, bytes) else value
    result = {
        "pid": os.getpid(),
        "options": options,
        "allow_hf32_matmul": torch.npu.matmul.allow_hf32,
        "allow_hf32_conv": torch.npu.conv.allow_hf32,
        "readback_api": "torch_npu._C._npu_getOption",
    }
    result["matches_requested"] = (
        options == EXPECTED_OPTIONS
        and result["allow_hf32_matmul"] is False
        and result["allow_hf32_conv"] is False
    )
    return result


def configure_npu_precision() -> dict:
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal():
        raise ValueError("Precision configuration requires one explicit visible chip")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_option(dict(EXPECTED_OPTIONS))
    result = read_npu_precision()
    if not result["matches_requested"]:
        raise RuntimeError(f"NPU precision option readback disagrees: {result}")
    return result


def configure_spawned_worker() -> None:
    directory = os.environ.get("AICS_NPU_PRECISION_DIR")
    if not directory:
        return
    result = configure_npu_precision()
    path = Path(directory) / f"precision-{os.getpid()}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def install_spawned_worker_hooks(*additional_hooks) -> None:
    """Defer device work until the TokenSpeed worker finishes spawn bootstrap.

    ``__mp_main__`` executes before the process target is unpickled. Setting
    ACL options there makes later platform detection initialize CANN's graph
    compiler while Python still forbids spawning its helper Manager process.
    Installing this stdlib-only wrapper is safe at that point; BaseProcess.run
    executes after bootstrap, before run_event_loop constructs the model.

    The exact target filter excludes CANN Managers and compiler helpers that
    also inherit the main module and environment.
    """
    from multiprocessing.process import BaseProcess

    original_run = BaseProcess.run

    @wraps(original_run)
    def run_after_bootstrap(process):
        target = getattr(process, "_target", None)
        if (
            getattr(target, "__module__", "") == "tokenspeed.runtime.engine.event_loop"
            and getattr(target, "__name__", "") == "run_event_loop"
        ):
            configure_spawned_worker()
            for hook in additional_hooks:
                hook()
        return original_run(process)

    BaseProcess.run = run_after_bootstrap


def collect_worker_precision(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("precision-*.json"))
    ]
