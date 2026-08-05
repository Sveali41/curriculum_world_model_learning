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

The world-model implementation is maintained in the separate
`Agent-Centric-Attentive-World-Model` repository. Curriculum code consumes
that repository and does not keep a second copy of `modelBased/` or the shared
`domain/` adapters.

For local development, install the sibling WM checkout:

```bash
pip install -r requirements-local-wm.txt
```

For a clean environment, `requirements.txt` installs the WM package from its
GitHub repository. Source `.env` before running experiments so that curriculum
paths and WM paths remain distinct.

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

## 1. Dataset Generation (Pre-requisite)

### Uniform Validation Data Collection 

To ensure fair evaluation and a rigorous comparison across all baselines, we first collect a unified, fixed uniform validation dataset. This guarantees that all reported metrics are evaluated on the exact same test distribution, eliminating variance from random dataset generation.

```bash
python modelBased/data/data_collect.py domain=crafter env.collect.data_type=uniform
python modelBased/data/data_collect.py domain=minigrid env.collect.data_type=uniform
python modelBased/data/data_collect.py domain=bipedalwalker env.collect.data_type=uniform
```

*Note: After generating the uniform datasets, ensure that the `validation_data_dir` parameter in your configurations (e.g., `modelBased/config/config.yaml` or under `trainer/conf/`) points to the resulting `*_uniform.npz` files before running the baselines.*

*(Optional) To collect standard random datasets or run standalone world-model/policy training, refer to the scripts in `modelBased/data/` and `modelBased/world_model/`.*

## 2. Main Experiments & Baselines

All experiment drivers use Hydra configs under [`trainer/conf/`](trainer/conf/). You can run any baseline by specifying the target `domain` and `seed`.

**1. MAC / adversarial curriculum generation (Ours)**
```bash
python trainer/mac_wm_learning.py domain=<domain> seed=0
```
*Ablations: append `ablation.type=no_history` or `ablation.type=no_diversity`.*

**2. DR baseline (domain randomization)**
```bash
python trainer/dr_baseline_experiment.py domain=<domain> seed=0
```

**3. Target-task baseline**
```bash
python trainer/target_baseline_experiment.py domain=<domain> seed=0
```

**4. P2E baseline**
```bash
python trainer/p2e_baseline.py domain=<domain> seed=0
```

**Multi-seed execution:**
For statistical significance across multiple seeds, use our provided shell runner:
```bash
bash trainer/run_seeds.sh
```

## 3. Evaluation and Logged Outputs

Evaluation is seamlessly integrated into the experiment drivers. During training, the scripts periodically validate on the uniform datasets generated in Step 1 and automatically write the metrics to CSV logs (e.g., `trainer/logs/results/`). 

Please refer to the **Reproducibility Table** below for the exact mapping between each experiment command and its specific output artifact.

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
