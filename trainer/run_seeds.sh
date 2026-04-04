#!/bin/bash
# Run DR baseline experiment for seeds 1-5
# Seed 0 already completed

cd /home/siyao/phd_file/Research/rlPractice/Curriculum_world_model_learning

for SEED in 0 1 2 3 4; do
    echo "==========================================="
    echo "Running DR Baseline with Seed=$SEED"
    echo "==========================================="
    python trainer/MAC_wm_learning.py seed=$SEED
    echo "[Done] Seed $SEED finished."
    echo ""
done

echo "All seeds (1-5) completed!"
