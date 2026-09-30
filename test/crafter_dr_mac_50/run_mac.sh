#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment_dir="$repo_root/test/crafter_dr_mac_50"
cd "$repo_root"
export TRAINER_ROOT="$repo_root"
export PROJECT_ROOT="$repo_root"

python_bin="${PYTHON:-python3}"
dry_run=0
run_mode=quick
extra_overrides=()
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --resume-full) run_mode=resume ;;
    *) extra_overrides+=("$1") ;;
  esac
  shift
done

if [[ "$run_mode" == resume ]]; then
  total_iterations=60
  force_fresh_start=false
  resume_training=true
else
  total_iterations=40
  force_fresh_start=true
  resume_training=false
fi

for seed in 0 1 2 3 4; do
  command=("$python_bin" -u -m trainer.mac_wm_learning
    --config-dir "$experiment_dir/conf"
    --config-name config_mac_crafter_dr_mac_50
    "seed=$seed"
    "domains.crafter.focal_gamma=1.0"
    "generator_agent.total_iterations=$total_iterations"
    "force_fresh_start=$force_fresh_start"
    "resume_training=$resume_training"
    "PPO.update_every_rounds=1"
    "mac_quick_run_id=mac_balanced_ewc20_epoch10_gamma1_seed${seed}"
    "${extra_overrides[@]}")
  printf '%q ' "${command[@]}"
  printf '\n'
  if (( !dry_run )); then
    "${command[@]}"
  fi
done
