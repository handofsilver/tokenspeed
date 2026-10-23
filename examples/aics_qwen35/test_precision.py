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

import os
import tempfile
import unittest
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from precision import (
    EXPECTED_OPTIONS,
    collect_worker_precision,
    configure_npu_precision,
    configure_spawned_worker,
    install_spawned_worker_hooks,
)


class TestPrecision(unittest.TestCase):
    def fake_modules(self, *, accept_options=True):
        state = {}
        npu = SimpleNamespace(
            matmul=SimpleNamespace(allow_hf32=True),
            conv=SimpleNamespace(allow_hf32=True),
        )

        def set_option(options):
            if accept_options:
                state.update(options)
                npu.matmul.allow_hf32 = state.get("ALLOW_MATMUL_HF32") != "disable"
                npu.conv.allow_hf32 = state.get("ALLOW_CONV_HF32") != "disable"

        npu.set_option = set_option
        extension = SimpleNamespace(
            _npu_getOption=lambda name: state.get(name, "").encode()
        )
        return {
            "torch": SimpleNamespace(npu=npu),
            "torch_npu": SimpleNamespace(_C=extension),
        }

    def test_configures_and_reads_actual_values(self):
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "7"}), patch.dict(
            "sys.modules", self.fake_modules()
        ):
            result = configure_npu_precision()
        self.assertTrue(result["matches_requested"])
        self.assertEqual(result["options"], EXPECTED_OPTIONS)
        self.assertEqual(result["pid"], os.getpid())

    def test_ignored_options_fail_instead_of_recording_request_as_effective(self):
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "7"}), patch.dict(
            "sys.modules", self.fake_modules(accept_options=False)
        ):
            with self.assertRaisesRegex(RuntimeError, "readback disagrees"):
                configure_npu_precision()

    def test_spawn_worker_persists_its_own_pid_readback(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"ASCEND_RT_VISIBLE_DEVICES": "7", "AICS_NPU_PRECISION_DIR": directory},
        ), patch.dict("sys.modules", self.fake_modules()):
            configure_spawned_worker()
            rows = collect_worker_precision(Path(directory))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pid"], os.getpid())
            self.assertTrue(rows[0]["matches_requested"])

    def test_multiple_chips_rejected_before_import_or_configuration(self):
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaisesRegex(ValueError, "one explicit visible chip"):
                configure_npu_precision()

    def test_spawn_install_defers_configuration_until_model_worker_runs(self):
        events = []

        def original_run(process):
            events.append("model_target")
            return 42

        target = SimpleNamespace(
            __module__="tokenspeed.runtime.engine.event_loop",
            __name__="run_event_loop",
        )
        process = SimpleNamespace(_target=target)
        with patch.object(BaseProcess, "run", original_run), patch(
            "precision.configure_spawned_worker",
            side_effect=lambda: events.append("precision"),
        ):
            install_spawned_worker_hooks(lambda: events.append("sampling"))
            self.assertEqual(events, [])
            self.assertEqual(BaseProcess.run(process), 42)
        self.assertEqual(events, ["precision", "sampling", "model_target"])

    def test_cann_manager_and_unrelated_targets_never_configure_npu(self):
        targets = [
            SimpleNamespace(
                __module__="multiprocessing.managers", __name__="_run_server"
            ),
            SimpleNamespace(__module__="unrelated", __name__="run_event_loop"),
            SimpleNamespace(
                __module__="tokenspeed.runtime.engine.event_loop", __name__="other"
            ),
            None,
        ]
        with patch.object(BaseProcess, "run", return_value="target_result"), patch(
            "precision.configure_spawned_worker"
        ) as configure:
            install_spawned_worker_hooks()
            for target in targets:
                with self.subTest(target=target):
                    result = BaseProcess.run(SimpleNamespace(_target=target))
                    self.assertEqual(result, "target_result")
            configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
