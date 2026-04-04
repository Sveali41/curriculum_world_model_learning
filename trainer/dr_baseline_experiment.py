import os
import sys
import tempfile
import torch
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, open_dict
from pathlib import Path
import glob

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
    Periodically validates on the fixed Target Tasks (20 for MiniGrid, 20 for Crafter).
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
    temp_data_dir = Path(cfg.dr_temp_data_dir)
    os.makedirs(temp_data_dir, exist_ok=True)
    summary_csv_path = log_dir / f"dr_summary_{domain_name}.csv"
    file_exists = summary_csv_path.exists()

    old_params, fisher = None, None
    csv_columns = [
        "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
        "gen_val_val_inv_loss", "gen_val_val_ce_loss", "gen_val_avg_val_loss_wm",
        "target_val_val_inv_loss", "target_val_val_ce_loss", "target_val_avg_val_loss_wm",
        "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
    ]

    # 2. Main Loop
    for iteration in range(cfg.generator_agent.total_iterations):
        print(f"\n>>> DR Iteration {iteration + 1}/{cfg.generator_agent.total_iterations}")
        
        # A. Collect Data from Random Maps
        # (generator.step uses agent_type='random' for both map generation and action)
        trajs = generator.step(old_params=old_params, iteration=iteration)
        
        # GeneratorInterface.step() returns a 9-tuple:
        # (.., raw_loss, ce_loss, inv_loss, div_reward, valid_trajs, solved_count, avg_bfs)
        if isinstance(trajs, tuple) and len(trajs) >= 7:
            gen_val_avg_val_loss_wm = float(trajs[2])
            gen_val_val_ce_loss = float(trajs[3])
            gen_val_val_inv_loss = float(trajs[4])
            gen_div_reward = float(trajs[5])
            valid_trajs = trajs[6]
            solvable_count = int(trajs[7]) if len(trajs) > 7 else 0
            avg_path_len = float(trajs[8]) if len(trajs) > 8 else 0.0
            if not isinstance(valid_trajs, list):
                valid_trajs = []
        else:
            gen_val_avg_val_loss_wm = 0.0
            gen_val_val_ce_loss = 0.0
            gen_val_val_inv_loss = 0.0
            gen_div_reward = 0.0
            valid_trajs = []
            solvable_count = 0
            avg_path_len = 0.0

        print(f"  [Generator] Collected {len(valid_trajs)} valid trajectories.")

        if not valid_trajs:
            print("  [Skip] No valid trajectories collected.")
            continue

        new_batch = convert_trajectories_to_batch(valid_trajs)
        current_transitions = len(new_batch["obs"]) if new_batch is not None and new_batch.get("obs") is not None else 0
        print(f"  [Data] Current batch transitions: {current_transitions}")
        
        # B. Train World Model
        if iteration % cfg.generator_agent.wm_train_frequency == 0:
            print("  [Training] Updating World Model...")
            replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None
            replay_size = len(replay_data["obs"]) if replay_data is not None and replay_data.get("obs") is not None else 0
            print(f"  [Training] Replay transitions: {replay_size}")

            # Align DR training path with MAC:
            # write the current batch to a standard a-h npz and train from cfg.data_dir.
            old_freeze = cfg.attention_model.freeze_weight
            old_data_dir = cfg.attention_model.data_dir
            temp_npz_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f"dr_{domain_name}_training_set_iter_{iteration}_",
                    suffix=".npz",
                    dir=str(temp_data_dir),
                    delete=False,
                ) as tmp_f:
                    temp_npz_path = tmp_f.name

                save_dict = {
                    'a': new_batch['obs'],
                    'b': new_batch['obs_next'],
                    'c': new_batch['act'],
                }
                if new_batch.get('rew') is not None:
                    save_dict['d'] = new_batch['rew']
                if new_batch.get('done') is not None:
                    save_dict['e'] = new_batch['done']
                if new_batch.get('info') is not None:
                    save_dict['f'] = new_batch['info']
                if new_batch.get('inv') is not None:
                    save_dict['g'] = new_batch['inv']
                if new_batch.get('inv_next') is not None:
                    save_dict['h'] = new_batch['inv_next']

                np.savez_compressed(temp_npz_path, **save_dict)
                cfg.attention_model.data_dir = temp_npz_path
                cfg.attention_model.freeze_weight = False

                # [FIX] Clear stale hooks to prevent ReferenceError: weakly-referenced object no longer exists
                if hasattr(wm, "_state_dict_hooks"):
                    wm._state_dict_hooks.clear()
                if hasattr(wm, "_parameters"):
                    for p_name, p in wm._parameters.items():
                        if p is not None and hasattr(p, "_hooks"):
                            p._hooks.clear()

                res_train, fisher, _ = AttentionWM_training.train_api(
                    cfg, net=wm, old_params=old_params, fisher=fisher,
                    replay_data=replay_data
                )
                old_params = res_train["old_params"]
            finally:
                cfg.attention_model.freeze_weight = old_freeze
                cfg.attention_model.data_dir = old_data_dir
                if temp_npz_path and os.path.exists(temp_npz_path):
                    os.remove(temp_npz_path)

            # 5. Reload Clean Instance
            print("  [System] Reloading model from checkpoint to clear hooks...")
            ckpt_path = cfg.attention_model.model_save_path
            wm = AttentionWorldModel(cfg.attention_model).to(device)
            try:
                ckpt = torch.load(ckpt_path, weights_only=False)
                if 'state_dict' in ckpt:
                    wm.load_state_dict(ckpt['state_dict'])
                else:
                    wm.load_state_dict(ckpt)
            except Exception as e:
                print(f"  [Warning] Failed to reload model: {e}")
                if isinstance(old_params, dict):
                    wm.load_state_dict(old_params)
                else:
                    wm.load_state_dict(old_params.state_dict())
            generator.sync_world_model(wm.state_dict())

            
        # C. Validation on Target Tasks (aligned with MAC: validate every iter after warmup)
        warmup_iters = int(getattr(cfg.generator_agent, "warmup_iterations", 0))
        if iteration >= (warmup_iters - 1):
            print(f"  [Validation] Running zero-shot test on all {d_cfg.val_n_phases} targets...")
            sum_loss, sum_ce, sum_inv, valid_count = 0.0, 0.0, 0.0, 0
            
            # Fix: Disable WandB during validation to prevent Hook ReferenceError and run spam
            old_use_wandb = cfg.attention_model.use_wandb
            cfg.attention_model.use_wandb = False

            task_indices = range(d_cfg.start_idx, d_cfg.start_idx + d_cfg.val_n_phases)
            for v_idx in task_indices:
                v_task = f"{d_cfg.val_task_prefix}{v_idx}"
                v_file = os.path.join(d_cfg.val_data_path, f"{v_task}{d_cfg.val_suffix}")
                
                if not os.path.exists(v_file): continue
                
                # Zero-shot validation logic
                res_v = validate_on_target_task(cfg, wm, None, d_cfg.val_data_path, f"{v_task}{d_cfg.val_suffix}", VALID_TIMES=1)

                
                if res_v:
                    l_val = res_v.get('avg_val_loss_wm', res_v.get('best_loss', 0.0))
                    ce_val = res_v.get('terrain_loss', 0.0)
                    inv_val = res_v.get('inventory_loss', 0.0)
                    sum_loss += l_val
                    sum_ce += ce_val
                    sum_inv += inv_val
                    valid_count += 1

            cfg.attention_model.use_wandb = old_use_wandb

            if valid_count > 0:
                target_val_avg_val_loss_wm = sum_loss / valid_count
                target_val_val_ce_loss = sum_ce / valid_count
                target_val_val_inv_loss = sum_inv / valid_count
                print(f"    -> Results: Avg Loss = {target_val_avg_val_loss_wm:.5f}")
            else:
                target_val_avg_val_loss_wm = 0.0
                target_val_val_ce_loss = 0.0
                target_val_val_inv_loss = 0.0
        else:
            target_val_avg_val_loss_wm = 0.0
            target_val_val_ce_loss = 0.0
            target_val_val_inv_loss = 0.0

        # D. Buffer Archiving
        fisher_buffer.add_from_batch(
            new_batch, 
            current_sample_ratio=cfg.attention_model.current_sample_ratio,
            fisher_buffer_elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio
        )
        print(f"  [Buffer] Archived {current_transitions} transitions. Buffer Size: {len(fisher_buffer)}")
        row = {
            "Seed": seed,
            "Iter": iteration + 1,
            "Gen_Mean_Reward": 0.0,
            "Gen_Loss": 0.0,
            "Gen_Entropy": 0.0,
            "Gen_Div_Reward": gen_div_reward,
            "gen_val_val_inv_loss": gen_val_val_inv_loss,
            "gen_val_val_ce_loss": gen_val_val_ce_loss,
            "gen_val_avg_val_loss_wm": gen_val_avg_val_loss_wm,
            "target_val_val_inv_loss": target_val_val_inv_loss,
            "target_val_val_ce_loss": target_val_val_ce_loss,
            "target_val_avg_val_loss_wm": target_val_avg_val_loss_wm,
            "New_Data_Size": current_transitions,
            "Buffer_Size": len(fisher_buffer),
            "Solvable_Count": solvable_count,
            "Avg_Path_Len": avg_path_len,
        }
        pd.DataFrame([row], columns=csv_columns).to_csv(
            summary_csv_path,
            index=False,
            mode="a",
            header=not file_exists,
        )
        file_exists = True
        torch.cuda.empty_cache()

        # E. Cleanup Temporary Data
        data_save_dir = Path(getattr(cfg.env.collect, "data_folder", str(TRAINER_PATH / "data")))
        temp_files = glob.glob(str(data_save_dir / f"UED_Dual_iter{iteration}_b*.npz"))
        if temp_files:
            print(f"  [Cleanup] Deleting {len(temp_files)} temporary files for Iteration {iteration}...")
            for f in temp_files:
                try: os.remove(f)
                except: pass

    print(f"\n[DR DONE] Log: {summary_csv_path}")

if __name__ == "__main__":
    run_dr_baseline_experiment()
