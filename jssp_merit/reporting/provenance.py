"""Portable source snapshot, runtime metadata and output integrity manifest."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np
import pandas as pd


def snapshot_source(destination: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "jssp_merit").rglob("*.py"))
    paths += [root / name for name in ("main.py", "requirements.txt", "pyproject.toml", "README.md")]
    paths += [root / "MILP_post_processing" / side / "milp_neighborhood.py"
              for side in ("machine", "jigs")]
    paths += [root / "MILP_post_processing" / "__init__.py"]
    paths += [root / "MILP_post_processing" / side / "__init__.py"
              for side in ("machine", "jigs")]
    paths += [root / "requirements-hybrid.txt"]
    hashes = {}
    with zipfile.ZipFile(destination / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                content = path.read_bytes()
                archive.writestr(relative, content)
                hashes[relative] = hashlib.sha256(content).hexdigest()
    metadata = {
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "processor": platform.processor(),
        "source_sha256": hashes,
    }
    (destination / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_manifest(destination: Path) -> None:
    entries = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append({"file": path.relative_to(destination).as_posix(),
                            "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    (destination / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
