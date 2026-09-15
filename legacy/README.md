# Legacy files

This directory contains inactive files retained for reproducibility or recovery.
Nothing under `legacy/` is part of the current training or evaluation entrypoints.

- `trainer/common/support.py` is an unreferenced helper from the earlier data-collection workflow.
- `artifacts/unexpanded_WORLD_MODEL_PATH/` contains checkpoints formerly written to the
  literal `trainer/${WORLD_MODEL_PATH}` path when the environment variable was not expanded.
  The checkpoint files are intentionally ignored by Git.

