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

"""Downloader tests with small synthetic payloads and an actual local HTTP server."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from examples.aics_qwen35_tp2.download_models import (
    download_file,
    load_models,
    transfer_entry,
    validate_entry,
)


class PayloadServer(ThreadingHTTPServer):
    def __init__(self, data: bytes, mode: str):
        self.data = data
        self.mode = mode
        self.request_ranges: list[str | None] = []
        super().__init__(("127.0.0.1", 0), PayloadHandler)


class PayloadHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = cast(PayloadServer, self.server)
        requested = self.headers.get("Range")
        server.request_ranges.append(requested)
        offset = (
            int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
        )
        if requested and server.mode != "ignore-range":
            self.send_response(206)
            start = offset + 1 if server.mode == "bad-range" else offset
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(server.data) - 1}/{len(server.data)}",
            )
        else:
            offset = 0
            self.send_response(200)
        body = server.data[offset:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if server.mode == "truncate-once" and len(server.request_ranges) == 1:
            self.wfile.write(body[: len(body) // 2])
            self.close_connection = True
        else:
            self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.payload = bytes(range(256)) * 1024
        self.server: PayloadServer | None = None
        self.thread: threading.Thread | None = None

    def tearDown(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join()
        self.temporary.cleanup()

    def start_server(self, mode: str) -> dict:
        self.server = PayloadServer(self.payload, mode)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return {
            "path": "asset.bin",
            "size": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "url": f"http://127.0.0.1:{self.server.server_port}/asset.bin",
        }

    def test_download_publishes_only_verified_file(self) -> None:
        entry = self.start_server("range")
        target = self.root / "asset.bin"
        result = download_file(entry, target, attempts=1, timeout=2)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse(target.with_name("asset.bin.partial").exists())

    def test_resume_uses_partial_content(self) -> None:
        entry = self.start_server("range")
        target = self.root / "asset.bin"
        target.with_name("asset.bin.partial").write_bytes(self.payload[:12345])
        download_file(entry, target, attempts=1, timeout=2)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(self.server.request_ranges, ["bytes=12345-"])

    def test_ignored_range_restarts_without_duplicate_prefix(self) -> None:
        entry = self.start_server("ignore-range")
        target = self.root / "asset.bin"
        target.with_name("asset.bin.partial").write_bytes(self.payload[:12345])
        download_file(entry, target, attempts=1, timeout=2)
        self.assertEqual(target.read_bytes(), self.payload)

    def test_interrupted_response_is_resumed(self) -> None:
        entry = self.start_server("truncate-once")
        target = self.root / "asset.bin"
        download_file(entry, target, attempts=2, timeout=2)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(
            self.server.request_ranges, [None, f"bytes={len(self.payload) // 2}-"]
        )

    def test_invalid_range_is_not_appended(self) -> None:
        entry = self.start_server("bad-range")
        target = self.root / "asset.bin"
        partial = target.with_name("asset.bin.partial")
        partial.write_bytes(self.payload[:12345])
        with self.assertRaises(RuntimeError):
            download_file(entry, target, attempts=1, timeout=2)
        self.assertFalse(target.exists())
        self.assertEqual(partial.read_bytes(), self.payload[:12345])

    def test_incorrect_hash_does_not_publish(self) -> None:
        entry = self.start_server("range")
        entry["sha256"] = "0" * 64
        target = self.root / "asset.bin"
        with self.assertRaises(RuntimeError):
            download_file(entry, target, attempts=1, timeout=2)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name("asset.bin.partial").exists())

    def test_existing_file_is_hashed_and_never_overwritten(self) -> None:
        entry = self.start_server("range")
        target = self.root / "asset.bin"
        corrupt = b"x" * len(self.payload)
        target.write_bytes(corrupt)
        with self.assertRaises(ValueError):
            download_file(entry, target, attempts=1, timeout=2)
        self.assertEqual(target.read_bytes(), corrupt)
        self.assertEqual(self.server.request_ranges, [])

    def test_existing_verified_file_avoids_network(self) -> None:
        entry = self.start_server("range")
        target = self.root / "asset.bin"
        target.write_bytes(self.payload)
        result = download_file(entry, target, attempts=1, timeout=2)
        self.assertEqual(result["status"], "existing")
        self.assertEqual(self.server.request_ranges, [])

    def test_manifest_paths_cannot_escape_destination(self) -> None:
        for path in ("/asset.bin", "../asset.bin", "model/../../asset.bin", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_entry({"path": path, "size": 1, "sha256": "0" * 64})

    def test_manifest_rejects_mutable_revision_url(self) -> None:
        entry = {
            "path": "config.json",
            "size": 1,
            "sha256": "0" * 64,
            "url": "https://huggingface.co/Qwen/example/resolve/main/config.json",
        }
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": [
                        {
                            "name": "example",
                            "repository": "Qwen/example",
                            "revision": "a" * 40,
                            "files": [entry],
                        }
                    ],
                }
            )
        )
        with self.assertRaises(ValueError):
            load_models(manifest, ["example"])

    def test_mirror_preserves_pinned_revision_and_expected_hash(self) -> None:
        entry = {
            "path": "model.safetensors",
            "size": 123,
            "sha256": "b" * 64,
            "url": f"https://huggingface.co/Qwen/example/resolve/{'a' * 40}/model.safetensors",
        }
        mirrored = transfer_entry(entry, "https://hf-mirror.com")
        self.assertEqual(
            mirrored["url"],
            f"https://hf-mirror.com/Qwen/example/resolve/{'a' * 40}/model.safetensors",
        )
        self.assertEqual(mirrored["size"], entry["size"])
        self.assertEqual(mirrored["sha256"], entry["sha256"])
        self.assertTrue(entry["url"].startswith("https://huggingface.co/"))

    def test_modelscope_source_requires_identical_content_and_fixed_revision(
        self,
    ) -> None:
        entry = {
            "path": "model.safetensors",
            "size": 123,
            "sha256": "b" * 64,
            "url": f"https://huggingface.co/Qwen/example/resolve/{'a' * 40}/model.safetensors",
            "alternative_sources": [
                {
                    "endpoint": "https://modelscope.cn",
                    "revision": "c" * 40,
                    "url": f"https://modelscope.cn/models/Qwen/example/resolve/{'c' * 40}/model.safetensors",
                    "size": 123,
                    "sha256": "b" * 64,
                }
            ],
        }
        selected = transfer_entry(entry, "https://modelscope.cn")
        self.assertEqual(selected["url"], entry["alternative_sources"][0]["url"])
        self.assertEqual(selected["canonical_url"], entry["url"])
        self.assertEqual(selected["sha256"], entry["sha256"])
        self.assertIsNone(selected["source_fallback"])
        entry["alternative_sources"][0]["sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            transfer_entry(entry, "https://modelscope.cn")

    def test_modelscope_missing_identical_asset_falls_back_to_hf_mirror(self) -> None:
        entry = {
            "path": "LICENSE",
            "size": 123,
            "sha256": "b" * 64,
            "url": f"https://huggingface.co/Qwen/example/resolve/{'a' * 40}/LICENSE",
        }
        selected = transfer_entry(entry, "https://modelscope.cn")
        self.assertEqual(selected["download_source_choice"], "https://modelscope.cn")
        self.assertEqual(selected["download_endpoint"], "https://hf-mirror.com")
        self.assertEqual(
            selected["url"],
            f"https://hf-mirror.com/Qwen/example/resolve/{'a' * 40}/LICENSE",
        )
        self.assertIsNotNone(selected["source_fallback"])


if __name__ == "__main__":
    unittest.main()
