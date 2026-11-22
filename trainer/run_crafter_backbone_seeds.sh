#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
job_dir="${1:-$repo/outputs/runs/crafter_backbone_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$job_dir")"
mkdir "$job_dir"  # Refuse to overwrite or mix with an existing experiment.
job_dir="$(cd "$job_dir" && pwd)"

# Preserve the executable source/config, even if the server checkout is dirty.
mkdir "$job_dir/provenance"
git rev-parse HEAD > "$job_dir/provenance/trainer.commit"
git -C wm rev-parse HEAD > "$job_dir/provenance/wm.commit"
for tree in . wm; do
    label=trainer
    [[ "$tree" != wm ]] || label=wm
    git -C "$tree" ls-files --cached --others --exclude-standard -z +        -- '*.py' '*.sh' '*.yaml' '*.yml' '*.txt' |
        sort -zu |
        tar -C "$tree" --null --no-recursion -T - -czf "$job_dir/provenance/$label-source.tar.gz"
done
for target in {1..20}; do
    sha256sum "trainer/data/crafter/target_tasks/crafter_target_task_${target}_uniform.npz"
done > "$job_dir/provenance/target_sha256.txt"
python -m pip freeze > "$job_dir/provenance/pip-freeze.txt"

printf 'Crafter backbone results: %s\n' "$job_dir"
for seed in 0 1 2 3 4; do
    seed_dir="$job_dir/seed$seed"
    mkdir "$seed_dir"
    printf 'Starting seed %s: %s/train.log\n' "$seed" "$seed_dir"
    python -u -m trainer.mac_wm_learning --config-name=config_crafter_backbone_ablation \
        "seed=$seed" "crafter_backbone_run_dir=$seed_dir" > "$seed_dir/train.log" 2>&1
done
