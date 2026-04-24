#!/bin/bash
# Run DR baseline experiment for seeds 1-5
# Seed 0 already completed

cd /home/siyao/phd_file/Research/rlPractice/Curriculum_world_model_learning

for SEED in 1 2 3 4; do
    echo "==========================================="
    echo "Running MAC WM Learning with Seed=$SEED"
    echo "==========================================="
    python trainer/p2e_baseline.py seed=$SEED
    echo "[Done] Seed $SEED finished."
    echo ""
done

echo "All seeds (1-5) completed!"
