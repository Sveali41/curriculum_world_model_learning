#!/bin/bash
set -euo pipefail
# Run P2E and Target Baseline experiments for seeds 1-5
# Seed 0 is usually the initial test seed

cd "$(cd "$(dirname "$0")/.." && pwd)"

for SEED in 0 1 2 ; do
    echo "=========================================================="
    echo "PHASE: Running Seed $SEED"
    echo "=========================================================="
    
    # 1. Run Plan-to-Explore Baseline
    # echo "[P2E] Starting exploration and training..."
    # python trainer/p2e_baseline.py seed=$SEED
    
    # 2. Run Target Random Baseline
    # echo "[Target] Starting random baseline collection and training..."
    # python3 trainer/target_baseline_experiment.py seed=$SEED

    # 3. run dr baseline experiment
    echo "[DR] Starting random baseline collection and training..."
    python trainer/dr_baseline_experiment.py seed=$SEED

    # 4. run mac baseline experiment
    # echo "[MAC] Starting random baseline collection and training..."
    # python trainer/mac_wm_learning.py seed=$SEED
    
    # echo "----------------------------------------------------------"
    # echo "[Done] Seed $SEED experiments completed."
    # echo "----------------------------------------------------------"
    # echo ""
done

# for SEED in 0 1 2 3 4; do
#     echo "=========================================================="
#     echo "PHASE: Running Seed $SEED"
#     echo "=========================================================="
    
#     # 1. Run Plan-to-Explore Baseline
#     echo "[P2E] Starting exploration and training..."
#     python3 trainer/p2e_baseline.py seed=$SEED
    
#     # 2. Run Target Random Baseline
#     # echo "[Target] Starting random baseline collection and training..."
#     # python trainer/target_baseline_experiment.py seed=$SEED

#     # # # 3. run dr baseline experiment
#     # echo "[DR] Starting random baseline collection and training..."
#     # python trainer/dr_baseline_experiment.py seed=$SEED

#     # 4. run mac baseline experiment
#     echo "[MAC] Starting random baseline collection and training..."
#     python trainer/mac_wm_learning.py seed=$SEED
    
#     echo "----------------------------------------------------------"
#     echo "[Done] Seed $SEED experiments completed."
#     echo "----------------------------------------------------------"
#     echo ""
# done



# ====== Crafter Ablation Experiments ======
# Ablation 1: MAC without Diversity Reward (no_diversity)
# for SEED in 0 1 2 3 4; do
#     echo "=========================================================="
#     echo "ABLATION: no_diversity | Seed $SEED"
#     echo "=========================================================="
#     python trainer/mac_wm_learning.py seed=$SEED ablation.type=no_diversity
# done

# Ablation 2: MAC without History Encoder (no_history)
# for SEED in 0 1 2 3 4; do
#     echo "=========================================================="
#     echo "ABLATION: no_history | Seed $SEED"
#     echo "=========================================================="
#     python trainer/mac_wm_learning.py seed=$SEED ablation.type=no_history
# done


echo "All experiments for seeds 1-5 have been successfully completed!"
