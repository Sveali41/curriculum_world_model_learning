"""Canonical filesystem locations for curriculum run artifacts.

Datasets and reusable checkpoints intentionally remain under ``trainer/data``
and ``wm/modelBased/models``.  Ephemeral run artifacts all live below the one
workspace-level ``outputs`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path


# Trainer run artifacts are anchored to this repository's outer workspace,
# independently of WM_ROOT.  Entry points set TRAINER_ROOT explicitly; the
# source-tree fallback also protects imports made before that setup.
WORKSPACE_ROOT = Path(
    os.environ.get("TRAINER_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
OUTPUTS_ROOT = WORKSPACE_ROOT / "outputs"
RESULTS_ROOT = OUTPUTS_ROOT / "results"
VISUALIZATIONS_ROOT = OUTPUTS_ROOT / "visualizations"
