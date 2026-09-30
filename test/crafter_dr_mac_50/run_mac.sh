#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment_dir="$repo_root/test/crafter_dr_mac_50"
cd "$repo_root"
export TRAINER_ROOT="$repo_root"
export PROJECT_ROOT="$repo_root"

python_bin="${PYTHON:-python3}"
dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
for seed in 0 1 2 3 4; do
  command=("$python_bin" -u -m trainer.mac_wm_learning
    --config-dir "$experiment_dir/conf"
    --config-name config_mac_crafter_dr_mac_50
    "seed=$seed"
    "domains.crafter.focal_gamma=1.0"
    "PPO.update_every_rounds=1"
    "mac_quick_run_id=mac_balanced_ewc20_epoch10_gamma1_seed${seed}"
    "$@")
  printf '%q ' "${command[@]}"
  printf '\n'
  if (( !dry_run )); then
    "${command[@]}"
  fi
done
