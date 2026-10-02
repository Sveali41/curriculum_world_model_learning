#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_root="$repo_root/test/bipedal_dr/seed0"
mkdir -p "$run_root"
cd "$repo_root"

python -u trainer/dr_baseline_experiment.py \
  domain=bipedalwalker seed=0 \
  "paths.outputs=$run_root" \
  "attention_model.model_save_path=$run_root/models/AttentionWM/dr_attention_world_model_bipedalwalker_none_lp_g1_seed0.ckpt" \
  "domains.bipedalwalker.data_folder=$run_root/data/" \
  "dr_temp_data_dir=$run_root/data/dr/" \
  "+attention_model.metrics_dir=$run_root/metrics/" \
  2>&1 | tee "$run_root/train.log"
