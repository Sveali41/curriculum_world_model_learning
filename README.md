# Research Code Release

This repository contains code for curriculum generation and world-model learning across three domains:

- `crafter`
- `minigrid`
- `bipedalwalker`

The repository is organized to follow common research-code release best practices: dependency specification, training code, evaluation code, reproducible commands, and a clear mapping from scripts to outputs.

Reference guideline: Papers with Code, "Tips for Publishing Research Code"  
https://github.com/paperswithcode/releasing-research-code

## Overview

The codebase contains four main experiment families:

1. `MAC` / adversarial curriculum generation with a learned world model
2. `DR` baseline (domain randomization / random generator)
3. `Target` baseline (train directly on fixed target-task datasets)
4. `P2E` baseline (Plan2Explore-style exploration with disagreement)

It also includes standalone code for:

- dataset collection
- world model training
- policy training / testing

## Repository Structure

```text
domain/                         environment-specific wrappers and utilities
generator/                      curriculum / generator models and interfaces
modelBased/
  data/                         data collection and dataloaders
  world_model/                  Attention-based world model training
  policy_training/              PPO and planning utilities
trainer/
  conf/                         Hydra experiment configs
  level/                        task definitions for each domain
  *.py                          main experiment entry points
requirements.txt                Python dependencies
.env                            local path configuration
```

## Environment Setup

### 1. Create a Python environment

`requirements.txt` is the main dependency specification for this repository.

Example setup with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

### 2. Set repository paths

Source the repo-local environment file from the repository root:

```bash
source .env
```

This sets:

- `PROJECT_ROOT`
- `TRAINER_PATH`
- `WORLD_MODEL_PATH`
- `TRAIN_DATASET_PATH`
- `MODEL_FPATH`

### 3. Notes on platform-specific packages

Some dependencies are environment-specific or GPU-specific, especially:

- `KNN-CUDA`
- CUDA-enabled `torch` / `torch-geometric` related packages
- simulation packages such as `mujoco`, `Box2D`, and `crafter`

If you are reproducing only a subset of experiments, you may need to adapt those packages to your local CUDA / OS setup.

## Main Entry Points

### Standalone dataset collection

Collect a dataset using the default config in `modelBased/config/config.yaml`:

```bash
python modelBased/data/data_collect.py domain=crafter
python modelBased/data/data_collect.py domain=minigrid
python modelBased/data/data_collect.py domain=bipedalwalker
```

### Standalone world model training

Train the world model from the collected dataset:

```bash
python modelBased/world_model/AttentionWM_training.py domain=crafter
python modelBased/world_model/AttentionWM_training.py domain=minigrid
python modelBased/world_model/AttentionWM_training.py domain=bipedalwalker
```

### Standalone policy training / testing

```bash
python modelBased/policy_training/PPO_world_training.py domain=minigrid
python modelBased/policy_training/PPO_world_test.py domain=minigrid
```

## Main Experiments

All experiment drivers use Hydra configs under [`trainer/conf/`](trainer/conf/).

### 1. MAC / curriculum generation

Config: `trainer/conf/config_mac.yaml`

```bash
python trainer/mac_wm_learning.py domain=crafter seed=0
python trainer/mac_wm_learning.py domain=minigrid seed=0
python trainer/mac_wm_learning.py domain=bipedalwalker seed=0
```

Ablations:

```bash
python trainer/mac_wm_learning.py domain=crafter seed=0 ablation.type=no_history
python trainer/mac_wm_learning.py domain=crafter seed=0 ablation.type=no_diversity
```

### 2. DR baseline

Config: `trainer/conf/config_dr.yaml`

```bash
python trainer/dr_baseline_experiment.py domain=crafter seed=0
python trainer/dr_baseline_experiment.py domain=minigrid seed=0
python trainer/dr_baseline_experiment.py domain=bipedalwalker seed=0
```

### 3. Target-task baseline

Config: `trainer/conf/config_target_baseline.yaml`

```bash
python trainer/target_baseline_experiment.py domain=crafter seed=0
python trainer/target_baseline_experiment.py domain=minigrid seed=0
python trainer/target_baseline_experiment.py domain=bipedalwalker seed=0
```

### 4. P2E baseline

Config: `trainer/conf/config_p2e.yaml`

```bash
python trainer/p2e_baseline.py domain=crafter seed=0
python trainer/p2e_baseline.py domain=minigrid seed=0
python trainer/p2e_baseline.py domain=bipedalwalker seed=0
```

### 5. Multi-seed runs

An example shell runner is provided in [`trainer/run_seeds.sh`](trainer/run_seeds.sh).

```bash
bash trainer/run_seeds.sh
```

Before using it, verify which experiment blocks are commented/uncommented.

## Evaluation and Logged Outputs

Evaluation is integrated into the main experiment scripts. During training, the scripts periodically validate on held-out or target-task datasets and write CSV logs.

Typical output locations:

- `trainer/logs/results/` for MAC
- `trainer/logs/results_dr/` for DR
- `trainer/logs/results_target_baseline/` for target baselines
- `trainer/logs/results_p2e/` for P2E

Representative output files:

- `trainer/logs/results/<domain>_ued_results_mask*.csv`
- `trainer/logs/results_dr/dr_summary_<domain>_mask*.csv`
- `trainer/logs/results_target_baseline/target_baseline_<domain>_mask*.csv`

These CSVs are the primary artifacts used to reproduce result tables and plots.

## Reproducibility Table

The table below links each experiment family to the exact command and the expected output log file.

| Setting | Domain flag | Command | Main output |
| --- | --- | --- | --- |
| MAC | `domain=crafter` | `python trainer/mac_wm_learning.py domain=crafter seed=0` | `trainer/logs/results/crafter_ued_results_mask*.csv` |
| MAC | `domain=minigrid` | `python trainer/mac_wm_learning.py domain=minigrid seed=0` | `trainer/logs/results/minigrid_ued_results_mask*.csv` |
| MAC | `domain=bipedalwalker` | `python trainer/mac_wm_learning.py domain=bipedalwalker seed=0` | `trainer/logs/results/bipedalwalker_ued_results*.csv` |
| DR | `domain=crafter` | `python trainer/dr_baseline_experiment.py domain=crafter seed=0` | `trainer/logs/results_dr/dr_summary_crafter_mask*.csv` |
| DR | `domain=minigrid` | `python trainer/dr_baseline_experiment.py domain=minigrid seed=0` | `trainer/logs/results_dr/dr_summary_minigrid_mask*.csv` |
| DR | `domain=bipedalwalker` | `python trainer/dr_baseline_experiment.py domain=bipedalwalker seed=0` | `trainer/logs/results_dr/dr_summary_bipedalwalker_mask*.csv` |
| Target baseline | `domain=crafter` | `python trainer/target_baseline_experiment.py domain=crafter seed=0` | `trainer/logs/results_target_baseline/target_baseline_crafter_mask*.csv` |
| Target baseline | `domain=minigrid` | `python trainer/target_baseline_experiment.py domain=minigrid seed=0` | `trainer/logs/results_target_baseline/target_baseline_minigrid_mask*.csv` |
| Target baseline | `domain=bipedalwalker` | `python trainer/target_baseline_experiment.py domain=bipedalwalker seed=0` | `trainer/logs/results_target_baseline/target_baseline_bipedalwalker_mask*.csv` |
| P2E | `domain=crafter` | `python trainer/p2e_baseline.py domain=crafter seed=0` | `trainer/logs/results_p2e/` |
| P2E | `domain=minigrid` | `python trainer/p2e_baseline.py domain=minigrid seed=0` | `trainer/logs/results_p2e/` |
| P2E | `domain=bipedalwalker` | `python trainer/p2e_baseline.py domain=bipedalwalker seed=0` | `trainer/logs/results_p2e/` |

## Reporting Final Results

For a fuller public release, we recommend adding a small result table here with the final metrics copied from the generated CSV files, for example:

| Domain | Method | Metric | Value |
| --- | --- | --- | --- |
| Crafter | MAC | target validation loss | `TBD` |
| Crafter | DR | target validation loss | `TBD` |
| MiniGrid | MAC | target validation loss | `TBD` |
| BipedalWalker | MAC | target validation loss | `TBD` |

Commands and output locations are provided above. Final numbers can be filled in from the exact CSV logs used for reporting.

## Pretrained Models

Pretrained checkpoints are not bundled in this code package.

Reason:

- checkpoint files are large
- serialized checkpoints may contain machine-specific metadata
- keeping them external simplifies repository maintenance

Recommended post-review release options, consistent with the Papers with Code guide:

- Zenodo
- GitHub Releases
- Hugging Face Hub

If pretrained weights are released later, this README should be updated with:

- a download link
- checksum
- exact evaluation command
- expected metric

## Important Notes for Reproducibility

- All major scripts use Hydra. Command-line overrides such as `domain=...`, `seed=...`, and `ablation.type=...` are expected.
- Paths are derived from `.env`; source it before running experiments.
- The repository contains domain-specific level files under `trainer/level/`.
- Large generated artifacts such as logs, checkpoints, temporary datasets, and notebooks are intentionally excluded from version control.

## Checklist Mapping

This repository currently covers the main ML code completeness items from the Papers with Code release guideline:

- Dependency specification: `requirements.txt`, `setup.py`
- Training code: provided
- Evaluation code: integrated into experiment drivers and standalone policy/world-model scripts
- Pretrained models: not bundled in this repository
- README with precise run commands: provided here

## Citation

If you use this repository in academic work, add the corresponding paper citation, project page, and checkpoint links here.
