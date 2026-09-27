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
for seed in 0 1; do
  command=("$python_bin" -u -m trainer.dr_baseline_experiment
    --config-dir "$experiment_dir/conf"
    --config-name config_dr_crafter_dr_mac_50
    "seed=$seed" "$@")
  printf '%q ' "${command[@]}"
  printf '\n'
  if (( !dry_run )); then
    "${command[@]}"
  fi
done
