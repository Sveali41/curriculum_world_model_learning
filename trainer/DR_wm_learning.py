import hydra
from omegaconf import DictConfig, open_dict
import os
import torch
import glob
import torch
import numpy as np
import copy
import csv

from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer

from generator.generator_interface import GeneratorInterface
from trainer.common.utils import (
    set_seed,
    validate_on_target_task,
    save_validation_csv,
    convert_trajectories_to_batch,
)


def filter_balanced_batch(new_batch, fisher_buffer, ratio=0.5, elements_ratio=0.5):
    """
    Filter the current batch using a balanced strategy (similar to Fisher Buffer).
    Prioritizes samples with key/door/lava based on elements_ratio.
    """
    if new_batch is None:
        return None

    try:
        obs_raw = new_batch['obs']
        total_len = len(obs_raw)
        total_quota = int(total_len * ratio)
        
        # If quota is valid, perform filtering
        if total_quota > 0 and total_len > 0:
            # Convert to tensor for mask calculation
            obs_tensor = torch.tensor(obs_raw) if not isinstance(obs_raw, torch.Tensor) else obs_raw
            if obs_tensor.device.type != 'cpu': 
                    obs_tensor = obs_tensor.cpu()
                    
            # Reuse helper from buffer instance
            near_elements_mask = fisher_buffer.get_agent_near_elements_mask(obs_tensor)
            near_indices_all = torch.where(near_elements_mask)[0].cpu().numpy()
            
            # Quota for special elements
            elements_quota = int(total_quota * elements_ratio)
            
            # Select Elements
            elements_selected = []
            if len(near_indices_all) > 0 and elements_quota > 0:
                pick_n = min(elements_quota, len(near_indices_all))
                elements_selected = np.random.choice(near_indices_all, pick_n, replace=False).tolist()
                
            # Select Random (Rest)
            remaining_quota = total_quota - len(elements_selected)
            total_indices = list(range(total_len))
            non_elements_pool = [i for i in total_indices if i not in elements_selected]
            
            if len(non_elements_pool) >= remaining_quota:
                    random_selected = np.random.choice(non_elements_pool, remaining_quota, replace=False).tolist()
            else:
                    random_selected = non_elements_pool 
                    
            # Combine
            all_selected_indices = elements_selected + random_selected
            np.random.shuffle(all_selected_indices)
            
            print(f"[Data Filter] Original: {total_len}, Filtered: {len(all_selected_indices)} "
                    f"(Ratio: {ratio}, Elements: {len(elements_selected)})")
            
            # Helper to slice
            def slice_batch(batch, inds):
                sliced = {}
                for k, v in batch.items():
                    if isinstance(v, (np.ndarray, list)):
                            sliced[k] = np.array(v)[inds]
                    elif torch.is_tensor(v):
                            sliced[k] = v[inds]
                    else:
                            sliced[k] = v 
                return sliced
                
            filtered_batch = slice_batch(new_batch, all_selected_indices)
            return filtered_batch
            
    except Exception as e:
        print(f"[Warning] Data filtering failed: {e}. Returning original batch.")
    
    return new_batch


@hydra.main(
    version_base=None,
    config_path=str(TRAINER_PATH / "conf"),
    config_name="config_UED",
)
def domain_randomization_baseline(cfg: DictConfig):
    """
    Domain Randomization (DR) Baseline.
    Uses 'RandomGeneratorAgent' instead of PPO.
    """

    # --------------------------------------
    # 1. 设置与初始化
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志与数据路径
    log_dir = TRAINER_PATH / "logs" / "results_dr" # [MODIFIED] Separate logs
    os.makedirs(log_dir, exist_ok=True)
    csv_path = log_dir / "dr_log.csv"
    
    # [METRIC] Suffix for CSV
    metric_suffix = ""
    if getattr(cfg.attention_model, "validation_metric", "mse") != "mse":
        metric_suffix = f"_{cfg.attention_model.validation_metric}"
        
    summary_csv_path = log_dir / f"experiment_summary_dr_mask_3_mse{metric_suffix}.csv"
    data_save_dir = TRAINER_PATH / "data"

    # === 初始化 Summary CSV ===
    with open(summary_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Iter", 
            "Gen_Mean_Reward",  # Initial Difficulty (Proxy)
            "Gen_Real_Loss",    # [NEW] Raw Loss on generated maps (Unscaled)
            "Gen_Loss", 
            "Gen_Entropy",      # [NEW] Entropy
            "Gen_Div_Reward",   # [NEW] Diversity Reward
            "WM_Val_Loss",      # Global Capability (Target Tasks Mean)
            "WM_Val_Max",       # [NEW] Worst-case Capability
            "WM_Val_Std",       # [NEW] Stability
            "Valid_Trajs",
            "New_Data_Size",
            "Buffer_Size",
            "Solvable_Count",
            "Avg_Path_Len"
        ])
    print(f"[Logger] Experiment summary will be saved to {summary_csv_path}")
    
    # === Set default mini_dataset_size for data collection ===
    with open_dict(cfg):
        # Default to 2000 steps minimum to ensure training stability
        if not hasattr(cfg.env.collect, "mini_dataset_size"):
            cfg.env.collect.mini_dataset_size = 3000
    
    print(f"[Config] Data collection mini_dataset_size: {cfg.env.collect.mini_dataset_size}")

    # === A. 初始化 World Model ===
    wm_instance = AttentionWorldModel(cfg.attention_model).to(device)

    # === B. 初始化 Generator Interface ===
    gen_interface = GeneratorInterface(
        world_model=wm_instance,
        device=device,
        cfg=cfg,
        agent_type='random' # [MODIFIED] Force Random Agent
    )

    # === C. 初始化 Fisher Replay Buffer ===
    fisher_buffer = FisherReplayBuffer(
        max_size=cfg.attention_model.fisher_buffer_size
    )

    # === D. 训练状态变量 ===
    old_params, fisher = None, None
    total_iterations = cfg.generator_agent.total_iterations
    # [MODIFIED] DR has no "Warmup" (Agent never learns). Force 0 to skip special logic.
    warmup_iterations = 0 
    wm_train_frequency = cfg.generator_agent.wm_train_frequency  

    # === E. 验证集定义 ===
    target_tasks = [
        "target_task0.txt",
        "target_task1.txt",
        "target_task2.txt",
        "target_task3.txt",
        "target_task4.txt",
        "target_task5.txt",
        "target_task6.txt",
        "target_task7.txt",
        "target_task8.txt",
        "target_task9.txt",
        "target_task10.txt",
        "target_task11.txt",
        "target_task12.txt",
        "target_task13.txt",
        "target_task14.txt",
        "target_task15.txt",
        "target_task16.txt",
        "target_task17.txt",
        "target_task18.txt",
        "target_task19.txt",
    ]

    target_files = [
        os.path.splitext(t)[0] + "_test_uniform.npz"
        for t in target_tasks
    ]

    print(
        f">>> Starting Domain Randomization (DR) Baseline for {total_iterations} iterations..."
    )

    # --------------------------------------
    # 2. 主循环 (The Loop)
    # --------------------------------------
    for iteration in range(total_iterations):
        print(
            f"\n=== Iteration {iteration + 1}/{total_iterations} ==="
        )

        # --------------------------------------------------------
        # Step 1: Generator 运行 (生成 -> 探索 -> 收集)
        # --------------------------------------------------------
        print(
            "[Generator] Generating environments and collecting trajectories..."
        )

        _, _, gen_raw_loss, gen_div_reward, valid_trajectories, solvable_count, avg_bfs_dist = gen_interface.step(iteration)

        num_valid_trajs = len(valid_trajectories)
        print(
            f"[Generator] Collected {num_valid_trajs} valid trajectories."
        )

        if num_valid_trajs == 0:
            print(
                "[Warning] No valid trajectories this round. But we still update Generator with failure penalties!"
            )
            # continue # FIX: Do NOT skip update! Let generator learn from failure (-5 reward).

        # --------------------------------------------------------
        # Step 2: 处理数据 (Prepare Buffer Input)
        # --------------------------------------------------------
        new_batch = convert_trajectories_to_batch(valid_trajectories)
        new_data_size = 0
        buffer_input = None  # Init for later use

        if new_batch is not None:
            new_data_size = len(new_batch['obs'])
            buffer_input = {
                "obs": new_batch["obs"],
                "obs_next": new_batch["obs_next"],
                "act": new_batch["act"],
                "info": new_batch["info"],
            }
            # Moved buffer update to AFTER training (Step 4.5) to prevent double counting


        # --------------------------------------------------------
        # Step 3: 更新 Generator (PPO Update)
        # --------------------------------------------------------
        # [MODIFIED] Warmup Fix: Do NOT update generator during warmup.
        # [MODIFIED] Warmup Fix: Do NOT update generator during warmup.
        if iteration >= warmup_iterations:
            gen_loss, gen_entropy, gen_mean_reward = gen_interface.update()

            if gen_loss is not None:
                print(
                    f"[Generator] Policy Updated. Loss: {gen_loss:.4f} | Entropy: {gen_entropy:.4f}"
                )
            else:
                print("[Generator] Policy Updated. Loss: NaN (Skipped due to instability)")
                gen_loss = 0.0 
                gen_entropy = 0.0
        else:
             print(f"[System] Warmup Phase ({iteration+1}/{warmup_iterations}): Skipping Generator Update.")
             gen_loss = 0.0
             gen_entropy = 0.0
             gen_mean_reward = 0.0

        # --------------------------------------------------------
        # Step 4: 更新 World Model (Adversarial Learning)
        # --------------------------------------------------------
        wm_final_loss = 0.0 # default if not trained
        
        # [MODIFIED] Warmup Fix: ALLOW WM training during warmup!
        if iteration % wm_train_frequency == 0:
            print("[World Model] Retraining on current + replay data...")
            
            # --- [New Filter Logic] ---
            # 1. Filter current batch (new_batch)
            if new_batch is not None:
                # [MODIFIED] Disabled filtering to match UED full-batch training
                # new_batch = filter_balanced_batch(
                #     new_batch, 
                #     fisher_buffer, 
                #     ratio=cfg.attention_model.current_sample_ratio, 
                #     elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio
                # )
                
                # Update logged data size (Not needed, already full size)
                # if new_batch is not None:
                #     new_data_size = len(new_batch['obs'])

                # Save filtered batch to disk
                current_data_path = data_save_dir / f"training_set_iter_{iteration}.npz"
                save_dict = {
                    'a': new_batch['obs'].cpu().numpy() if torch.is_tensor(new_batch['obs']) else new_batch['obs'],
                    'b': new_batch['obs_next'].cpu().numpy() if torch.is_tensor(new_batch['obs_next']) else new_batch['obs_next'],
                    'c': new_batch['act'].cpu().numpy() if torch.is_tensor(new_batch['act']) else new_batch['act'],
                    'f': new_batch['info']
                }
                np.savez_compressed(current_data_path, **save_dict)
                cfg.attention_model.data_dir = str(current_data_path)
            
            # --- [Replay Data Logic] ---
            # 2. Get historical data from buffer
            if len(fisher_buffer) > 0:
                replay_data = fisher_buffer.export_dict()
            else:
                replay_data = None
                print("[System] Fisher Buffer is empty. Training on current batch only.")

            # 3. Handle model freezing/reloading (Old Params Logic)
            if old_params is None:
                # First time training or after warmup
                pass # Already handled by init
            
            # Unfreeze for training
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = False 
            for param in wm_instance.parameters():
                param.requires_grad = True

            # 4. Train WM
            old_params, fisher, _ = AttentionWM_training.train_api(
                cfg,
                wm_instance, 
                old_params,
                fisher,
                replay_data=replay_data,
            )
            
            # Log loss
            # Note: Need to extract loss from somewhere, train_api prints it but doesn't return scalar easily unless parsed
            # For logging, we reuse validation loss later.
            
            # Restore freeze config
            cfg.attention_model.freeze_weight = old_freeze

            # 5. Reload Clean Instance
            print("[System] Reloading model from checkpoint to clear hooks...")
            ckpt_path = cfg.attention_model.model_save_path
            wm_instance = AttentionWorldModel(cfg.attention_model).to(device)
            try:
                # Fix for PyTorch 2.6 security change compatibility
                ckpt = torch.load(ckpt_path, weights_only=False)
                if 'state_dict' in ckpt:
                    wm_instance.load_state_dict(ckpt['state_dict'])
                else:
                    wm_instance.load_state_dict(ckpt)
            except Exception as e:
                print(f"[Warning] Failed to reload model: {e}. Using potentially dirty instance.")
                if isinstance(old_params, dict):
                     wm_instance.load_state_dict(old_params)
                else:
                     wm_instance.load_state_dict(old_params.state_dict())

            # Resync generator
            gen_interface.sync_world_model(wm_instance.state_dict())

            print(
                "[System] World Model updated, reloaded, and synced to Generator."
            )

        # --------------------------------------------------------
        # Step 4.5: Update Fisher Buffer (Archive Current Data)
        # --------------------------------------------------------
        # We do this AFTER training so that 'replay_data' (used in training) 
        # strictly contains PAST data, while 'curr_data' contains CURRENT data.
        if buffer_input is not None:
             fisher_buffer.update_combined(
                buffer_input,
                cfg.attention_model.current_sample_ratio,
                cfg.attention_model.fisher_buffer_elements_ratio,
            )
             print(
                f"[Buffer] Archived {new_data_size} transitions. "
                f"Buffer Size: {len(fisher_buffer)}"
            )
        
        # --------------------------------------------------------
        # Step 5: 写 CSV 日志
        # --------------------------------------------------------
        # Step 5: 验证与日志 (Validation on Target Tasks)
        # --------------------------------------------------------
        # Step 5: 验证与日志 (Validation on Target Tasks)
        # --------------------------------------------------------
        target_mean_loss = 0.0
        target_max_loss = 0.0
        target_std_loss = 0.0
        # --------------------------------------------------------
        # Step 5: 验证与日志 (Validation on Target Tasks)
        # --------------------------------------------------------
        target_mean_loss = 0.0
        target_max_loss = 0.0
        target_std_loss = 0.0
        
        # [MODIFIED] Always validate to show the full learning curve
        # Previously skippped during warmup
        if True: 
            # print("\n>>> Validating on Target Tasks...")
            avg_losses = []
            
            # Temporary set to validation mode
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = True

            # Fix: Disable WandB during validation to prevent Hook ReferenceError and run spam
            old_use_wandb = cfg.attention_model.use_wandb
            cfg.attention_model.use_wandb = False

            for t_name, t_file in zip(target_tasks, target_files):
                # Fix: Create a FRESH model instance for validation instead of deepcopying.
                
                loss = validate_on_target_task(
                    cfg, 
                    net=wm_instance, # Map 'wm' to 'net'
                    old_params=None, # pass default
                    data_save_dir=str(data_save_dir), # data_save_dir is defined earlier as TRAINER_PATH / "data"
                    target_file=t_file, 
                    phase_name=f"Iter_{iteration}",
                    VALID_TIMES=1
                )
                
                if loss is not None:
                     avg_losses.append(loss)
            
            # Restore configs
            cfg.attention_model.freeze_weight = old_freeze
            cfg.attention_model.use_wandb = old_use_wandb

            if avg_losses:
                target_mean_loss = sum(avg_losses) / len(avg_losses)
                target_max_loss = max(avg_losses)
                target_std_loss = np.std(avg_losses)
                
                print(f"[Metrics] Global Capability (Target Tasks) -> Loss: {target_mean_loss:.6f} | Max: {target_max_loss:.6f}")
            else:
                target_max_loss = 0.0
                target_std_loss = 0.0

        # --------------------------------------------------------
        # Step 6: 写 CSV 日志 (Experiment Summary)
        # --------------------------------------------------------
        if summary_csv_path is not None:
            try:
                with open(summary_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        iteration + 1,
                        f"{gen_mean_reward:.4f}",
                        f"{gen_raw_loss:.6f}",
                        f"{gen_loss:.4f}",
                        f"{gen_entropy:.4f}",
                        f"{gen_div_reward:.4f}",
                        f"{target_mean_loss:.6f}",
                        f"{target_max_loss:.6f}",
                        f"{target_std_loss:.6f}",
                        num_valid_trajs,
                        new_data_size,
                        len(fisher_buffer),
                        solvable_count,
                        f"{avg_bfs_dist:.2f}"
                    ])
            except Exception as e:
                print(f"[Error] Failed to write CSV log: {e}")

        # --------------------------------------------------------
        # Step 7: Cleanup Temporary Data
        # --------------------------------------------------------
        # Delete generated trajectory files for this iteration to save space
        temp_files = glob.glob(str(data_save_dir / f"UED_temp_data_path_iter{iteration}_*.npz"))
        if temp_files:
            print(f"[Cleanup] Deleting {len(temp_files)} temporary files for Iteration {iteration}...")
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"[Warning] Could not delete {f}: {e}")
            print(f"[Cleanup] Done.")

    print(">>> Domain Randomization Baseline Finished.")


if __name__ == "__main__":
    domain_randomization_baseline()
