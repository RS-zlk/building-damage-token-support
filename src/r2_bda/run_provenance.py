"""Run provenance utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": digest.hexdigest(),
    }


class RunProvenance:

    def __init__(
        self,
        output_dir: Path,
        args: Any,
        repo_root: Path,
        run_type: str,
        inputs: dict[str, Any],
        allow_existing: bool = False,
    ) -> None:
        self.output_dir = output_dir.resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not allow_existing:
            raise FileExistsError(
                f"Output directory is not empty: {self.output_dir}. "
                "Use a new --output-dir or set --allow-existing-output."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "run_metadata.json"
        command = [sys.executable, *sys.argv]
        (self.output_dir / "command.txt").write_text(
            f"cd {shlex.quote(str(Path.cwd()))}\n{shlex.join(command)}\n",
            encoding="utf-8",
        )
        config = {key: _json_value(value) for key, value in vars(args).items()}
        (self.output_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        status = _git(repo_root, "status", "--short")
        self.metadata = {
            "schema_version": 1,
            "run_type": run_type,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "working_directory": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "CUDA_VISIBLE_DEVICES")
            },
            "git": {
                "commit": _git(repo_root, "rev-parse", "HEAD"),
                "branch": _git(repo_root, "branch", "--show-current"),
                "dirty": bool(status),
                "status_short": status or "",
            },
            "inputs": _json_value(inputs),
        }
        self._write()
        self._previous_excepthook = sys.excepthook

        def record_failure(exc_type, exc_value, traceback):
            self.fail(f"{exc_type.__name__}: {exc_value}")
            self._previous_excepthook(exc_type, exc_value, traceback)

        sys.excepthook = record_failure

    def _write(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def complete(self, **results: Any) -> None:
        self.metadata.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            results=_json_value(results),
        )
        self._write()
        sys.excepthook = self._previous_excepthook

    def fail(self, error: str) -> None:
        self.metadata.update(
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
        )
        self._write()
        sys.excepthook = self._previous_excepthook
