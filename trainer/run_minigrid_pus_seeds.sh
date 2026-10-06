#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

job_dir="${1:-$repo/outputs/runs/minigrid_pus_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$job_dir"

export TRAINER_ROOT="$repo"
export PROJECT_ROOT="$repo"
export TRAINER_PATH="$repo/trainer"
export GENERATOR_PATH="$repo/generator"
export WM_ROOT="$repo/wm"
export WORLD_MODEL_PATH="$repo/wm/modelBased"
export TRAIN_DATASET_PATH="$repo/wm/modelBased/data/train_world_model"
export ENV_PATH="$repo/level"

test -f generator/minigrid_pus.py || { echo "MiniGrid PUS code is missing" >&2; exit 1; }
python3 -c 'import hydra, torch, minigrid' || {
    echo "Run this script inside the training Apptainer environment" >&2
    exit 1
}

echo "PUS job directory: $job_dir"
for seed in 0 1 2 3 4; do
    seed_dir="$job_dir/seed$seed"
    if [[ -e "$seed_dir/results/pus_summary_minigrid_mask5_focal_reservoir.csv" ]]; then
        echo "Refusing to overwrite existing PUS seed $seed results in $seed_dir" >&2
        exit 1
    fi
    mkdir -p "$seed_dir/results" "$seed_dir/models" "$seed_dir/temp"
    export MODEL_FPATH="$seed_dir/models"

    echo "START pus seed=$seed"
    python3 -u -m trainer.dr_baseline_experiment \
        --config-name=config_pus \
        domain=minigrid \
        seed="$seed" \
        domains.minigrid.exploration_policy=random \
        generator_agent.total_iterations=50 \
        generator_agent.warmup_iterations=0 \
        dr_log_dir="$seed_dir/results" \
        dr_temp_data_dir="$seed_dir/temp" \
        paths.outputs="$seed_dir/artifacts" \
        force_fresh_start=true \
        resume_training=false \
        > "$seed_dir/train.log" 2>&1
    echo "DONE pus seed=$seed"
done
