#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
set -a
source "$PROJECT_DIR/.env"
set +a

PYTHON_BIN="${PYTHON_BIN:-/home/siyao/Apps/anaconda3/envs/miniGrid/bin/python}"
SEED="${SEED:-0}"
ITERATIONS="${ITERATIONS:-12}"
TRANSITIONS="${TRANSITIONS:-800}"
WM_EPOCHS="${WM_EPOCHS:-2}"
TARGET_SAMPLES="${TARGET_SAMPLES:-200}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_DIR/outputs/results/diagnostics/minigrid_quick/$RUN_ID"

mkdir -p "$RUN_ROOT"/{logs,models,temp,results,environments}
printf '%s\n' "$RUN_ROOT" > "$PROJECT_DIR/outputs/results/diagnostics/minigrid_quick/latest_run.txt"

COMMON_OVERRIDES=(
  "seed=$SEED"
  "force_fresh_start=true"
  "generator_agent.total_iterations=$ITERATIONS"
  "domains.minigrid.iter_transition_budget=$TRANSITIONS"
  "domains.minigrid.max_edits_layout=0.3"
  "attention_model.n_epochs=$WM_EPOCHS"
  "attention_model.target_validation_max_samples=$TARGET_SAMPLES"
  "attention_model.debug_mode=false"
  "env.collect.save_env_visualize=true"
  "env.collect.save_coverage_visualize=false"
)

run_mac() {
  local name="$1"
  local ablation="$2"

  mkdir -p "$RUN_ROOT/results/$name" "$RUN_ROOT/temp/$name" "$RUN_ROOT/environments/$name"
  echo "[Quick Diagnostic] Starting $name"
  "$PYTHON_BIN" "$PROJECT_DIR/trainer/mac_wm_learning.py" \
    "${COMMON_OVERRIDES[@]}" \
    "ablation.type=$ablation" \
    "mac_results_dir=$RUN_ROOT/results/$name" \
    "mac_temp_data_dir=$RUN_ROOT/temp/$name" \
    "attention_model.model_save_path=$RUN_ROOT/models/$name.ckpt" \
    "env.collect.env_visualize_save_path=$RUN_ROOT/environments/$name" \
    2>&1 | tee "$RUN_ROOT/logs/$name.log"
}

run_dr() {
  local name="dr"
  mkdir -p "$RUN_ROOT/results/$name" "$RUN_ROOT/temp/$name" "$RUN_ROOT/environments/$name"
  echo "[Quick Diagnostic] Starting $name"
  "$PYTHON_BIN" "$PROJECT_DIR/trainer/dr_baseline_experiment.py" \
    "${COMMON_OVERRIDES[@]}" \
    "ablation.type=none" \
    "generator_agent.total_iterations=$ITERATIONS" \
    "dr_log_dir=$RUN_ROOT/results/$name" \
    "dr_temp_data_dir=$RUN_ROOT/temp/$name" \
    "attention_model.model_save_path=$RUN_ROOT/models/$name.ckpt" \
    "env.collect.env_visualize_save_path=$RUN_ROOT/environments/$name" \
    2>&1 | tee "$RUN_ROOT/logs/$name.log"
}

cd "$PROJECT_DIR"

run_dr
run_mac "mac_full" "none"
run_mac "mac_no_history" "no_history"
run_mac "mac_no_diversity" "no_diversity"

"$PYTHON_BIN" "$PROJECT_DIR/trainer/analyze_minigrid_quick_diagnostics.py" "$RUN_ROOT"

echo "[Quick Diagnostic] Complete: $RUN_ROOT"
