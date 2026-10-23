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

"""Download or verify pinned checkpoints on the Linux course server.

Example::

    python examples/aics_qwen35_tp2/download_models.py --mode download \
        --endpoint https://huggingface.co \
        --models Qwen3.5-4B Qwen3.5-9B Qwen3.5-27B --destination ../models

Interrupted transfers resume from ``.partial`` files. Existing final files are
verified and never overwritten when they differ from the manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path, PurePosixPath

CHUNK_SIZE = 8 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def emit(event: dict) -> None:
    with PRINT_LOCK:
        print(json.dumps(event, ensure_ascii=False), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def validate_entry(entry: dict) -> None:
    """Reject malformed paths and unverified assets before a transfer starts."""
    path = PurePosixPath(entry["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe manifest path: {entry['path']}")
    if not isinstance(entry["size"], int) or entry["size"] < 0:
        raise ValueError(f"Invalid size for {entry['path']}")
    if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError(f"Missing SHA256 for {entry['path']}")


def verify_file(path: Path, entry: dict) -> dict:
    """Check both size and SHA256, including files already present on disk."""
    size = path.stat().st_size
    if size != entry["size"]:
        raise ValueError(f"Size mismatch for {path}: {size} != {entry['size']}")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path}")
    return {"file": str(path), "bytes": size, "sha256": digest}


def transfer_once(entry: dict, temporary: Path, *, timeout: float) -> None:
    """Resume one transfer, restarting when the server ignores the Range header."""
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > entry["size"]:
        temporary.unlink()
        offset = 0
    headers = {"User-Agent": "tokenspeed-coursework-checkpoint-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(entry["url"], headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != offset or int(match[3]) != entry["size"]:
                raise ValueError(
                    f"Invalid Content-Range for {entry['path']}: {content_range}"
                )
            mode = "ab"
        elif response.status == 200:
            mode = "wb"
            offset = 0
        else:
            raise ValueError(f"Unexpected HTTP status {response.status}")
        downloaded = offset
        last_progress = time.monotonic()
        with temporary.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                downloaded += len(chunk)
                if downloaded > entry["size"]:
                    raise ValueError(
                        f"Response exceeds expected size for {entry['path']}"
                    )
                output.write(chunk)
                now = time.monotonic()
                if now - last_progress >= 30:
                    emit(
                        {
                            "file": str(temporary),
                            "status": "downloading",
                            "bytes": downloaded,
                            "total_bytes": entry["size"],
                        }
                    )
                    last_progress = now
            output.flush()
            os.fsync(output.fileno())


def download_file(entry: dict, target: Path, *, attempts: int, timeout: float) -> dict:
    """Verify an existing file or atomically publish a verified partial download."""
    validate_entry(entry)
    if target.exists():
        return {**verify_file(target, entry), "status": "existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if not temporary.exists() or temporary.stat().st_size != entry["size"]:
                emit(
                    {
                        "file": str(target),
                        "status": "transfer",
                        "attempt": attempt + 1,
                        "download_url": entry["url"],
                        "download_source_choice": entry.get("download_source_choice"),
                        "download_endpoint": entry.get("download_endpoint"),
                    }
                )
                transfer_once(entry, temporary, timeout=timeout)
            result = verify_file(temporary, entry)
            os.replace(temporary, target)
            return {**result, "file": str(target), "status": "verified"}
        except (OSError, ValueError, http.client.HTTPException) as error:
            last_error = error
            if temporary.exists() and temporary.stat().st_size == entry["size"]:
                # A full but corrupt partial cannot be repaired by appending bytes.
                temporary.unlink()
            emit(
                {
                    "file": str(target),
                    "status": "retry",
                    "attempt": attempt + 1,
                    "error": str(error),
                }
            )
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(
        f"Download failed after {attempts} attempts: {target}"
    ) from last_error


def load_models(manifest: Path, names: list[str]) -> list[dict]:
    data = json.loads(manifest.read_text())
    if data["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    available = {model["name"]: model for model in data["models"]}
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(
            f"Unknown models: {sorted(unknown)}; available: {sorted(available)}"
        )
    models = [available[name] for name in dict.fromkeys(names)]
    for model in models:
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"Unpinned revision for {model['name']}")
        for entry in model["files"]:
            validate_entry(entry)
            prefix = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/"
            if entry["url"] != f"{prefix}{entry['path']}":
                raise ValueError(f"Unpinned URL for {entry['path']}")
            for source in entry.get("alternative_sources", []):
                validate_alternative_source(entry, source, model["repository"])
    return models


def validate_alternative_source(entry: dict, source: dict, repository: str) -> None:
    if source.get("endpoint") != "https://modelscope.cn":
        raise ValueError("Unsupported alternative checkpoint source")
    revision = source.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Alternative source revision is not pinned")
    expected_url = (
        f"https://modelscope.cn/models/{repository}/resolve/{revision}/{entry['path']}"
    )
    if source.get("url") != expected_url:
        raise ValueError("Alternative source URL does not match its pinned revision")
    if source.get("size") != entry["size"] or source.get("sha256") != entry["sha256"]:
        raise ValueError("Alternative source differs from canonical checkpoint content")


def transfer_entry(entry: dict, endpoint: str) -> dict:
    """Select a pinned, content-identical source without changing the expected hash."""
    official_origin = "https://huggingface.co"
    if endpoint not in (
        official_origin,
        "https://hf-mirror.com",
        "https://modelscope.cn",
    ):
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    if not entry["url"].startswith(f"{official_origin}/"):
        raise ValueError(f"Expected official manifest URL: {entry['url']}")
    actual_endpoint = endpoint
    fallback = None
    if endpoint == "https://modelscope.cn":
        sources = entry.get("alternative_sources", [])
        repository = (
            entry["url"].removeprefix(f"{official_origin}/").split("/resolve/", 1)[0]
        )
        for source in sources:
            validate_alternative_source(entry, source, repository)
        if len(sources) > 1:
            raise ValueError("Ambiguous ModelScope alternatives")
        if sources:
            url = sources[0]["url"]
        else:
            actual_endpoint = "https://hf-mirror.com"
            fallback = "No content-identical pinned ModelScope asset in the manifest"
            url = f"{actual_endpoint}{entry['url'].removeprefix(official_origin)}"
    else:
        url = f"{endpoint}{entry['url'].removeprefix(official_origin)}"
    return {
        **entry,
        "url": url,
        "canonical_url": entry["url"],
        "download_source_choice": endpoint,
        "download_endpoint": actual_endpoint,
        "source_fallback": fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("download", "verify"), required=True)
    parser.add_argument(
        "--endpoint",
        choices=(
            "https://huggingface.co",
            "https://hf-mirror.com",
            "https://modelscope.cn",
        ),
        required=True,
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("model-manifest.json")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error(
            "Run checkpoint operations on the Linux course server, not the local computer."
        )
    if args.workers < 1 or args.attempts < 1 or args.timeout <= 0:
        parser.error("workers, attempts and timeout must be positive")
    models = load_models(args.manifest, args.models)
    entries = [
        (
            transfer_entry(entry, args.endpoint),
            args.destination / model["name"] / entry["path"],
        )
        for model in models
        for entry in model["files"]
    ]
    emit(
        {
            "status": "start",
            "mode": args.mode,
            "download_endpoint": args.endpoint,
            "download_source_choice": args.endpoint,
            "models": [model["name"] for model in models],
            "total_bytes": sum(model["total_bytes"] for model in models),
        }
    )

    def process(item: tuple[dict, Path]) -> dict:
        entry, target = item
        if args.mode == "verify":
            result = {**verify_file(target, entry), "status": "verified"}
        else:
            result = download_file(
                entry, target, attempts=args.attempts, timeout=args.timeout
            )
        return {
            **result,
            "canonical_url": entry["canonical_url"],
            "download_source_choice": entry["download_source_choice"],
            "download_endpoint": entry["download_endpoint"],
            "download_url": entry["url"],
            "source_fallback": entry["source_fallback"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process, entries):
            emit(result)
    emit({"status": "complete", "mode": args.mode, "files": len(entries)})


if __name__ == "__main__":
    main()
