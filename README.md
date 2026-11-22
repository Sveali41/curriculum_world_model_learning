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
generator/                      curriculum / generator models and interfaces
curriculum_domain/              curriculum-only domain extensions
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

For local development, initialize and install the WM submodule:

```bash
git submodule update --init --recursive
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
python -m modelBased.data.data_collect domain=crafter env.collect.data_type=uniform
python -m modelBased.data.data_collect domain=minigrid env.collect.data_type=uniform
python -m modelBased.data.data_collect domain=bipedalwalker env.collect.data_type=uniform
```

*Note: The shared WM configuration lives in the Agent-Centric repository under `modelBased/config/config.yaml`; update it or the relevant `trainer/conf/` configuration to point to the resulting `*_uniform.npz` files.*

*(Optional) To collect standard random datasets or run standalone world-model/policy training, use the installed WM package modules under `modelBased.data` and `modelBased.world_model`.*

## 2. Main Experiments & Baselines

All experiment drivers use Hydra configs under [`trainer/conf/`](trainer/conf/). You can run any baseline by specifying the target `domain` and `seed`.

**1. MAC / adversarial curriculum generation (Ours)**
```bash
python trainer/mac_wm_learning.py domain=<domain> seed=0
```
*Ablations: append `ablation.type=no_history` or `ablation.type=no_diversity`.*

For the Crafter prediction-error reward comparison, use
[`config_crafter_preloss.yaml`](trainer/conf/config_crafter_preloss.yaml):

```bash
python -m trainer.mac_wm_learning --config-name=config_crafter_preloss seed=0
```

This retains the saved LP/preloss training settings (including training
`focal_gamma=0`, EWC 20, and 50 WM updates after 10 warmup rounds), uses
`preloss` without reward normalization, and evaluates uniform targets 1–20
with 500 fixed samples per target and evaluation subset seed 0. Each training
seed has separate data, results, and checkpoints. Override
`crafter_preloss_run_dir` for an independent rerun. Existing checkpoints need
only reevaluation on this target protocol, not retraining. Report single-update
values or trailing-five-update averages consistently across methods; averaging
is a result aggregation step, not a training configuration option.

**Crafter architecture ablation: AC-AWM vs. CNN spatial backbone.**

```bash
bash trainer/run_crafter_backbone_seeds.sh
```

Run from the checkout with both the trainer and the nested WM changes present,
using the usual Python training environment. The launcher runs seeds 0–4
sequentially, writes separate logs/checkpoints per seed, and snapshots source,
dependencies and target archive hashes. Pass a new output directory as the
first argument if desired. To run one seed, use
[the paired preset](trainer/conf/config_crafter_backbone_ablation.yaml):

```bash
python -m trainer.mac_wm_learning --config-name=config_crafter_backbone_ablation \
  seed=0 crafter_backbone_run_dir=/absolute/run/seed0
```

To resume from an intact completed iteration, repeat the same command with
`resume_training=true force_fresh_start=false`.

Only AC-AWM drives the LP editor. Within a seed, the CNN arm uses the identical
current archive and replay input for each WM update, the same data selection
settings, and its own weights and EWC/Fisher state. Both use the same local
view, action and inventory inputs, and prediction heads; only the spatial
attention mixer is replaced by residual convolutional mixing. Both evaluate
on targets 1–20 using the same 500-sample subsets (seed 0), with 10 warmup
rounds and 50 WM updates. Compare `target_val_*` with
`cnn_target_val_*` in the per-seed summary CSV using the same transition
budgets. The run's timing includes the comparison arm; it is not an
AC-AWM-only compute measurement. This tests spatial attention on
MAC-collected data, not the effect of the agent-centric view itself.

MAC uses `generator_agent.warmup_iterations` rounds for generator diversity
warmup in Crafter, MiniGrid, and BipedalWalker, without collecting environment
transitions or LP probes. WM training and replay begin after warmup, followed
by normal collection and pre/post WM learning-progress evaluation.
The MiniGrid exploration policy also remains idle during warmup. Generator
warmup still uses compute.

For LP, each environment reuses a random 40% subset of its collected
transitions, without a separate probe rollout or exclusion from WM training
and replay. The same subset is evaluated before and after the WM update.
Sampling is without replacement, deterministic for the experiment seed,
iteration, and environment index, and rounds the sample count up (at least
one transition for a nonempty trajectory). The fraction is configured by
`generator_agent.learning_progress_sample_fraction`. DR does not sample LP
subsets or compute pre/post LP losses in any domain; its LP log fields are
left unavailable (NaN).

Generator diagnostics are written beside the MAC CSV (`*.manifest.json` and
`*.maps.jsonl`).

**Random-edit + LP baseline (MiniGrid and Crafter)**

```bash
python -m trainer.mac_wm_learning --config-name config_random_lp domain=minigrid seed=0
python -m trainer.mac_wm_learning --config-name config_random_lp_crafter seed=0
```

The [baseline configuration](trainer/conf/config_random_lp.yaml) inherits the
current MAC protocol, replacing PPO with the existing random editor and an
[LP environment pool](generator/random_lp.py). It is a PLR-inspired control,
not a reproduction of original PLR. The first batch contains only new maps;
later batches use `ceil(batch_size * new_fraction)` new maps, filling remaining
slots from the pool without replacement (or with more new maps if needed).
Defaults are capacity 32, new fraction 0.5, and uniform exploration mix 0.2.
Replay probabilities mix uniform sampling with normalized `max(pre - post, 0)`;
if all weights are zero, sampling is uniform. Scores use the latest visit only.
FIFO eviction follows insertion order, not LP or last-visit time.

Each visit recollects transitions from the stored layout, colors, states, and
initial inventory. The full batch enters ordinary WM training and transition
replay; its same 40% probe subsets measure pre/post LP. There are no separate
candidate rollouts or per-candidate WM updates. Invalid trajectories do not
enter the pool; revisited entries with invalid probes receive zero priority.
Crafter probes with no changed events have undefined changed-only loss: retain
these environments with zero LP priority and log `lp_valid=false` with null
loss/LP fields, excluding them from LP means. Nonfinite losses on probes with
changed events fail explicitly. Diversity is diagnostic only.
No PPO warmup is used: the default 50 rounds match MAC's 50 WM updates
after 10 warmup rounds. The [Crafter preset](trainer/conf/config_random_lp_crafter.yaml)
matches the historical Crafter WM settings, including EWC `lambda=20` and
slot-stratified Fisher sampling, but retains the current 40% in-rollout LP
measurement rather than collecting separate probe rollouts. It uses 8,000
training transitions per update (200k/400k at updates 25/50) and evaluates the
20 frozen `_uniform.npz` targets at 500 transitions per target with subset seed 0.

Artifacts are isolated under `outputs/runs/random_edit_lp/minigrid/seed<seed>/`
or `outputs/runs/random_edit_lp/crafter_matched/seed<seed>/` for the Crafter preset.
Use `random_lp_run_dir=/absolute/new/run` for an independent rerun. Map sidecars
record environment IDs, new/replayed status, conditional sampling probability,
score age, pre/post losses, and pool size. Iteration-boundary resume supports
Crafter and MiniGrid with the random explorer. It saves WM, LP pool, replay
(including reservoir admission count and RNG), diversity archives and global RNG
state. Append `resume_training=true force_fresh_start=false` with the original
run directory and configuration, increasing `generator_agent.total_iterations`
to the cumulative stopping iteration. For example, ten consecutive jobs stopping
at 5, 10, ..., 50 preserve fifty updates per seed. Each job must depend on its
same-seed predecessor's successful completion. Missing or inconsistent state
fails explicitly. MiniGrid's fresh environment RNG is currently entropy-seeded;
exact cross-run comparisons additionally require controlling that RNG in the
validation fixture. The production stochastic kernel is unchanged.
Focused validation: `python -m unittest tests.test_random_lp -v` (pool selection,
LP measurement wiring in both domains, generator-state restore and reservoir
continuation). Validated with 13 focused tests and CPU smoke runs using two environments,
64 transitions per round, and one WM epoch: two MiniGrid rounds, plus one
Crafter round followed by resume to round two. These checks verify execution
and restore behavior, not baseline performance or convergence.

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

Evaluation is seamlessly integrated into the experiment drivers. During training, the scripts periodically validate on the uniform datasets generated in Step 1 and automatically write metrics below `outputs/results/`.

Hydra run metadata and console logs for all curriculum and baseline entry points
are stored in the single workspace-level `outputs/` directory. World-model
entry points use the separate `wm/outputs/` directory, including planner,
policy-evaluation, coverage, and visualization artifacts. Both locations are
resolved absolutely, so running a command from `trainer/`, `wm/`, or the
workspace root does not create another nested `outputs/` directory.

The workspace output tree is organized as follows:

- `outputs/hydra/`: Hydra run logs and multiruns
- `outputs/results/`: MAC, DR, P2E, target-baseline, and continual-learning metrics
- `outputs/planning/`: retained MPC evaluation results
- `outputs/visualizations/`: dataset, environment, error, and model figures
- `outputs/wandb/`: local Weights & Biases run files

Historical experiments formerly stored in `test/` and `trainer/results/`
now live alongside these outputs. Bipedal runs, Crafter DR/MAC comparisons,
quick comparisons, and Bipedal contact focal ablations are under
`outputs/results/`. MiniGrid MPC sweeps are under
`outputs/planning/minigrid_mpc_k_sweep/`, and the former `test/results (1)/`
is under `outputs/planning/minigrid_hk_comparison/`. Their metrics,
checkpoints, datasets, and protocol records are retained. Saved run metadata
may still contain the original `test/` or `trainer/results/` paths; these historical records were
not rewritten.

Generated datasets stay under `trainer/data/`, and reusable model checkpoints
stay under `wm/modelBased/models/`; these are persistent inputs/artifacts rather
than run output.

Please refer to the **Reproducibility Table** below for the exact mapping between each experiment command and its specific output artifact.

## Reproducibility Table

The table below links each experiment family to the exact command and the expected output log file.

| Setting | Domain flag | Command | Main output |
| --- | --- | --- | --- |
| MAC | `domain=crafter` | `python trainer/mac_wm_learning.py domain=crafter seed=0` | `outputs/results/mac/mac_crafter_results.csv` |
| MAC | `domain=minigrid` | `python trainer/mac_wm_learning.py domain=minigrid seed=0` | `outputs/results/mac/minigrid_ued_results_mask*.csv` |
| MAC | `domain=bipedalwalker` | `python trainer/mac_wm_learning.py domain=bipedalwalker seed=0` | `outputs/results/mac/bipedalwalker_ued_results*.csv` |
| DR-EWC (balanced inventory) | `domain=crafter` | `python trainer/dr_baseline_experiment.py domain=crafter seed=0` | `outputs/results/dr/dr_crafter_results.csv` |
| DR | `domain=minigrid` | `python trainer/dr_baseline_experiment.py domain=minigrid seed=0` | `outputs/results/dr/dr_summary_minigrid_mask*.csv` |
| DR | `domain=bipedalwalker` | `python trainer/dr_baseline_experiment.py domain=bipedalwalker seed=0` | `outputs/results/dr/dr_summary_bipedalwalker_mask*.csv` |
| Target baseline | `domain=crafter` | `python trainer/target_baseline_experiment.py domain=crafter seed=0` | `outputs/results/target_baseline/target_baseline_crafter_mask*.csv` |
| Target baseline | `domain=minigrid` | `python trainer/target_baseline_experiment.py domain=minigrid seed=0` | `outputs/results/target_baseline/target_baseline_minigrid_mask*.csv` |
| Target baseline | `domain=bipedalwalker` | `python trainer/target_baseline_experiment.py domain=bipedalwalker seed=0` | `outputs/results/target_baseline/target_baseline_bipedalwalker_mask*.csv` |
| P2E | `domain=crafter` | `python trainer/p2e_baseline.py domain=crafter seed=0` | `outputs/results/p2e/` |
| P2E | `domain=minigrid` | `python trainer/p2e_baseline.py domain=minigrid seed=0` | `outputs/results/p2e/` |
| P2E | `domain=bipedalwalker` | `python trainer/p2e_baseline.py domain=bipedalwalker seed=0` | `outputs/results/p2e/` |

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
