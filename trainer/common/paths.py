"""Canonical filesystem locations for curriculum run artifacts.

Datasets and reusable checkpoints intentionally remain under ``trainer/data``
and ``wm/modelBased/models``.  Ephemeral run artifacts all live below the one
workspace-level ``outputs`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"
RESULTS_ROOT = OUTPUTS_ROOT / "results"
VISUALIZATIONS_ROOT = OUTPUTS_ROOT / "visualizations"

