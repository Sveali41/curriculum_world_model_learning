import os
import sys
import torch
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, open_dict
from pathlib import Path

# Add project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.world_model import AttentionWM_training
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from generator.generator_interface import GeneratorInterface
from trainer.common.utils import (
    set_seed, validate_on_target_task, convert_trajectories_to_batch
)

@hydra.main(version_base=None, config_path="conf", config_name="config_dr")
def run_dr_baseline_experiment(cfg: DictConfig):
    """
    Unified DR Baseline Experiment: Random maps -> Random actions -> Train WM.
    Periodically validates on the fixed Target Tasks (20 for MiniGrid, 6 for Crafter).
    """
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    domain_name = cfg.domain
    d_cfg = cfg.domains[domain_name]
    
    print(f"\n{'='*80}")
    print(f"### [DR BASEMENT START] Domain: {domain_name.upper()} | Seed: {seed}")
    print(f"{'='*80}\n")

    # Override Attention Model configs based on Domain
    with open_dict(cfg):
        cfg.attention_model.env_type = domain_name
        cfg.attention_model.grid_shape = d_cfg.grid_shape
        cfg.attention_model.obs_norm_values = d_cfg.obs_norm
        cfg.attention_model.action_norm_values = d_cfg.action_norm

    # 1. Initialization
    wm = AttentionWorldModel(cfg.attention_model).to(device)
    # DR usually generates random maps via GeneratorInterface (agent_type='random')
    generator = GeneratorInterface(wm, device, cfg, agent_type='random')
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    
    log_dir = Path(cfg.dr_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    summary_csv_path = log_dir / f"dr_summary_{domain_name}_{seed}.csv"
    
    all_results = []
    old_params, fisher = None, None

    # 2. Main Loop
    for iteration in range(cfg.generator_agent.total_iterations):
        print(f"\n>>> DR Iteration {iteration + 1}/{cfg.generator_agent.total_iterations}")
        
        # A. Collect Data from Random Maps
        # (generator.step uses agent_type='random' for both map generation and action)
        trajs = generator.step(old_params=old_params, iteration=iteration)
        
        # Handle returns of generator.step (UED interface vs DR interface)
        if isinstance(trajs, tuple) and len(trajs) > 0:
            # We assume it returned (valid_trajs, extra_stats...) 
            valid_trajs = trajs[0]
            if not isinstance(valid_trajs, list): valid_trajs = []
        else:
            valid_trajs = []

        if not valid_trajs:
            print("  [Skip] No valid trajectories collected.")
            continue

        new_batch = convert_trajectories_to_batch(valid_trajs)
        
        # B. Train World Model
        if iteration % cfg.generator_agent.wm_train_frequency == 0:
            print("  [Training] Updating World Model...")
            replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None
            
            res_train, fisher, _ = AttentionWM_training.train_api(
                cfg, net=wm, old_params=old_params, fisher=fisher, 
                replay_data=replay_data, direct_data=new_batch
            )
            old_params = res_train["old_params"]
            
        # C. Periodic Validation (Zero-shot test on all targets)
        if iteration % 5 == 0:
            print(f"  [Validation] Running zero-shot test on all {d_cfg.val_n_phases} targets...")
            sum_loss, valid_count = 0.0, 0
            phase_metrics = {"Iter": iteration + 1}
            
            task_indices = range(d_cfg.start_idx, d_cfg.start_idx + d_cfg.val_n_phases)
            for v_idx in task_indices:
                v_task = f"{d_cfg.val_task_prefix}{v_idx}"
                v_file = os.path.join(d_cfg.val_data_path, f"{v_task}{d_cfg.val_suffix}")
                
                if not os.path.exists(v_file): continue
                
                # Zero-shot validation logic
                res_v = validate_on_target_task(cfg, wm, None, d_cfg.val_data_path, f"{v_task}{d_cfg.val_suffix}", VALID_TIMES=1)
                
                if res_v:
                    l_val = res_v.get('avg_val_loss_wm', res_v.get('best_loss', 0.0))
                    sum_loss += l_val
                    valid_count += 1
                    phase_metrics[f"Task_{v_task}_Loss"] = l_val

            if valid_count > 0:
                phase_metrics["Avg_Loss"] = sum_loss / valid_count
                print(f"    -> Results: Avg Loss = {phase_metrics['Avg_Loss']:.5f}")
            
            all_results.append(phase_metrics)
            pd.DataFrame(all_results).to_csv(summary_csv_path, index=False)

        # D. Buffer Archiving
        fisher_buffer.add_from_batch(new_batch, current_sample_ratio=cfg.attention_model.ewc_ratio)
        torch.cuda.empty_cache()

    print(f"\n[DR DONE] Log: {summary_csv_path}")

if __name__ == "__main__":
    run_dr_baseline_experiment()
