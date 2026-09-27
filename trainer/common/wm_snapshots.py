"""Retain the exact Lightning checkpoint used after a Crafter WM update."""

import os
import shutil
import tempfile
from pathlib import Path


def save_wm_update_snapshot(checkpoint_path, run_dir, iteration):
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot save WM update snapshot; checkpoint is missing: {source}")
    snapshot_dir = Path(run_dir) / "wm_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    destination = snapshot_dir / f"iter_{int(iteration):03d}.ckpt"
    with tempfile.NamedTemporaryFile(dir=snapshot_dir, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
