import os
import sys
import torch
import numpy as np
import pandas as pd
import copy
import hydra
from omegaconf import DictConfig, open_dict
from pathlib import Path

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from modelBased.common.support import Support
from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.common.utils import TRAINER_PATH
from trainer.common.utils import set_seed

@hydra.main(version_base=None, config_path="conf", config_name="config_cl")
def run_target_baseline_experiment(cfg: DictConfig):
    """
    Unified Baseline Script: Trains World Model directly on Target Tasks data.
    Supports both Crafter (6 tasks) and MiniGrid (20 tasks).
    """
    # 1. Setup & Environment
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    domain_name = cfg.domain
    d_cfg = cfg.domains[domain_name]
    
    print(f"\n{'='*80}")
    print(f"### [START SESSION] Domain: {domain_name.upper()} | Seed: {seed}")
    print(f"{'='*80}\n")

    # Override Attention Model parameters based on Domain
    with open_dict(cfg):
        cfg.attention_model.env_type = domain_name
        cfg.attention_model.grid_shape = d_cfg.grid_shape
        cfg.attention_model.obs_norm_values = d_cfg.obs_norm
        cfg.attention_model.action_norm_values = d_cfg.action_norm

    # 2. Initialization
    net = AttentionWorldModel(cfg.attention_model).to(device)
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    
    old_params = None
    fisher = None
    all_results = []
    
    log_dir = TRAINER_PATH / "logs" / "results_target_baseline"
    os.makedirs(log_dir, exist_ok=True)
    csv_out_path = log_dir / f"target_baseline_{domain_name}_{seed}.csv"

    # 3. Main Phase Loop (Iterating through tasks)
    n_phases = d_cfg.n_phases
    task_indices = range(d_cfg.start_idx, d_cfg.start_idx + n_phases)
    task_names = [f"{d_cfg.task_prefix}{i}" for i in task_indices]

    for i, current_task in enumerate(task_names):
        train_path = os.path.join(d_cfg.data_path, f"{current_task}{d_cfg.train_suffix}")
        
        print(f"\n[PHASE {i+1}/{n_phases}] Training on: {current_task}")
        if not os.path.exists(train_path):
            print(f"Warning: Dataset {train_path} not found. Skipping phase.")
            continue

        # Load data and slice for granular training (optional, following crafter_baseline logic)
        full_data = np.load(train_path, allow_pickle=True)
        data_keys = list(full_data.keys())
        total_len = len(full_data[data_keys[0]])
        
        sub_step_size = 8000
        n_sub_steps = max(1, total_len // sub_step_size)
        
        for sub_idx in range(n_sub_steps):
            print(f"  -> Sub-Step {sub_idx+1}/{n_sub_steps} (Iter: {i+1})")
            
            start_i = sub_idx * sub_step_size
            end_i = min((sub_idx + 1) * sub_step_size, total_len)
            sub_data = {k: full_data[k][start_i:end_i] for k in data_keys}

            # --- A. Training Step ---
            replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None
            train_res, fisher, net = AttentionWM_training.train_api(
                cfg, net=net, old_params=old_params, fisher=fisher, 
                replay_data=replay_data, direct_data=sub_data
            )
            old_params = train_res["old_params"]

            # --- B. Validation Step (on ALL targets) ---
            if sub_idx == n_sub_steps - 1 or sub_idx % 2 == 0:
                print(f"    [Validating] Testing on all {n_phases} target tasks...")
                phase_metrics = {"Phase": f"{i+1}.{sub_idx+1}", "Trained_On": current_task}
                
                sum_total = 0.0
                valid_count = 0

                for v_idx, v_task in enumerate(task_names):
                    val_file = os.path.join(d_cfg.data_path, f"{v_task}{d_cfg.val_suffix}")
                    if not os.path.exists(val_file): continue
                    
                    val_cfg = copy.deepcopy(cfg)
                    val_cfg.attention_model.freeze_weight = True
                    val_cfg.attention_model.data_dir = val_file
                    
                    v_res, _, _ = AttentionWM_training.train_api(val_cfg, net=net, old_params=None, fisher=None)
                    
                    # Extract Metrics
                    metrics = v_res.get("avg_val_loss", {})
                    if isinstance(metrics, list) and len(metrics) > 0: metrics = metrics[0]
                    
                    l_val = float(metrics.get('avg_val_loss_wm', metrics.get('best_loss', 0.0)))
                    sum_total += l_val
                    valid_count += 1
                    phase_metrics[f"Task_{v_task}_Loss"] = l_val
                    
                if valid_count > 0:
                    phase_metrics["Avg_Loss"] = sum_total / valid_count
                    print(f"    -> Results: Avg Loss = {phase_metrics['Avg_Loss']:.5f}")
                
                all_results.append(phase_metrics)
                pd.DataFrame(all_results).to_csv(csv_out_path, index=False)

        # 4. Finalize Phase: Archive to Replay Buffer
        print(f"  [Buffer] Archiving {current_task} data...")
        fisher_buffer.add_from_npz(train_path, current_sample_ratio=cfg.attention_model.ewc_ratio)
        torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Experiment Complete. CSV saved to: {csv_out_path}")

if __name__ == "__main__":
    run_target_baseline_experiment()
