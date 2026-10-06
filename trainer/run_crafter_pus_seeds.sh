#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$script_dir"
while [[ ! -f "$repo/trainer/dr_baseline_experiment.py" ]]; do
    parent="$(dirname "$repo")"
    if [[ "$parent" == "$repo" ]]; then
        echo "Could not find the project root above $script_dir" >&2
        exit 1
    fi
    repo="$parent"
done

if [[ "${1:-}" == "--prepare" ]]; then
    job_dir="${2:-$repo/outputs/runs/crafter_pus_ordinary_replay_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$job_dir"
    job_dir="$(cd "$job_dir" && pwd)"
    if [[ -e "$job_dir/run.sh" ]]; then
        echo "Run script already exists in $job_dir" >&2
        exit 1
    fi
    cp "${BASH_SOURCE[0]}" "$job_dir/run.sh"
    printf '%s\n' "$job_dir"
    exit 0
fi

if [[ "$script_dir" == "$repo/trainer" ]]; then
    echo "Prepare a result directory first: bash trainer/run_crafter_pus_seeds.sh --prepare" >&2
    exit 2
fi

job_dir="$script_dir"
cd "$repo"

export TRAINER_ROOT="$repo"
export PROJECT_ROOT="$repo"
export TRAINER_PATH="$repo/trainer"
export GENERATOR_PATH="$repo/generator"
export WM_ROOT="$repo/wm"
export WORLD_MODEL_PATH="$repo/wm/modelBased"
export TRAIN_DATASET_PATH="$repo/wm/modelBased/data/train_world_model"
export ENV_PATH="$repo/level"

test -f generator/crafter_pus.py || { echo "Crafter PUS code is missing" >&2; exit 1; }

echo "Crafter PUS results: $job_dir"
for seed in 0 1 2 3 4; do
    seed_dir="$job_dir/seed$seed"
    if [[ -e "$seed_dir/train.log" || -e "$seed_dir/results/pus_crafter_results.csv" || -e "$seed_dir/results/pus_selected_settings.csv" ]]; then
        echo "Refusing to overwrite seed $seed results in $seed_dir" >&2
        exit 1
    fi
    mkdir -p "$seed_dir/results" "$seed_dir/models" "$seed_dir/temp"
    export MODEL_FPATH="$seed_dir/models"

    echo "START crafter PUS seed=$seed"
    if python3 -u -m trainer.dr_baseline_experiment \
        --config-name=config_crafter_pus \
        seed="$seed" \
        dr_log_dir="$seed_dir/results" \
        dr_temp_data_dir="$seed_dir/temp" \
        paths.outputs="$seed_dir/artifacts" \
        force_fresh_start=true \
        resume_training=false \
        > "$seed_dir/train.log" 2>&1; then
        echo "DONE crafter PUS seed=$seed"
    else
        status=$?
        echo "FAILED crafter PUS seed=$seed (exit $status): $seed_dir/train.log" >&2
        exit "$status"
    fi
done
