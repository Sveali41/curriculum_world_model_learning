import sys
import os
ROOT_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import hydra
from omegaconf import DictConfig, open_dict
from pathlib import Path
import torch
import glob
import torch
import numpy as np
import copy
import math

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


def _safe_int_cfg(value, default=0, name="value"):
    if value is None:
        print(f"[Config Warning] {name} is None, fallback to {default}.")
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[Config Warning] {name}={value} is invalid, fallback to {default}.")
        return int(default)


def _apply_domain_collection_budget(cfg: DictConfig, env_type: str):
    """
    Configure per-rollout collection size from domain-level per-iteration budget.
    effective_per_rollout = ceil(iter_transition_budget / generator_batch_size)
    """
    domain_cfg = cfg.domains[env_type] if hasattr(cfg, "domains") and env_type in cfg.domains else None
    if domain_cfg is None:
        print(f"[Config Warning] Missing domains.{env_type}; keep existing collection settings.")
        return

    with open_dict(cfg):
        batch_override = getattr(domain_cfg, "generator_batch_size", None)
        if batch_override is not None:
            cfg.generator_agent.batch_size = _safe_int_cfg(
                batch_override,
                default=getattr(cfg.generator_agent, "batch_size", 8),
                name=f"domains.{env_type}.generator_batch_size",
            )

    batch_size = max(
        1,
        _safe_int_cfg(
            getattr(cfg.generator_agent, "batch_size", 8),
            default=8,
            name="generator_agent.batch_size",
        ),
    )
    iter_budget = _safe_int_cfg(
        getattr(domain_cfg, "iter_transition_budget", None),
        default=max(
            1,
            _safe_int_cfg(
                getattr(cfg.env.collect, "maximum_dataset_size", 500),
                default=500,
                name="env.collect.maximum_dataset_size",
            ) * batch_size,
        ),
        name=f"domains.{env_type}.iter_transition_budget",
    )
    per_rollout_max = max(1, int(math.ceil(iter_budget / batch_size)))

    with open_dict(cfg):
        cfg.env.collect.maximum_dataset_size = per_rollout_max
        cfg.env.collect.mini_dataset_size = per_rollout_max

    expected_total = per_rollout_max * batch_size
    print(
        f"[Config] Domain budget applied | domain={env_type} | "
        f"iter_budget={iter_budget} | batch_size={batch_size} | "
        f"per_rollout_max={per_rollout_max} | expected_iter_total={expected_total}"
    )


@hydra.main(
    version_base=None,
    config_path=str(TRAINER_PATH / "conf"),
    config_name="config_mac",
)
def adversarial_ued_training(cfg: DictConfig):
    """
    UED Adversarial Training Loop.
    Integrates Generator (PPO), World Model (AttentionWM), and Continual Learning (Fisher Buffer).
    """

    # --------------------------------------
    # 1. 设置与初始化
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import csv

    # 日志与数据路径
    log_dir = TRAINER_PATH / "logs" / "results"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = log_dir / "ued_adversarial_log.csv"
    # [ABLATION] Suffix for CSV
    ablation_suffix = ""
    if hasattr(cfg, "ablation") and cfg.ablation.type != "none":
        ablation_suffix = f"_{cfg.ablation.type}"
    
    # [METRIC] Suffix for CSV (if not default mse)
    metric_suffix = ""
    if getattr(cfg.attention_model, "validation_metric", "mse") != "mse":
        metric_suffix = f"_{cfg.attention_model.validation_metric}"
        
    # Always construct path (handles empty suffixes naturally for default case)
    env_type = getattr(cfg.attention_model, "env_type", "minigrid")
    mask_suffix = f"_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"

    summary_csv_path = log_dir / f"{env_type}_ued_results{mask_suffix}{ablation_suffix}{metric_suffix}.csv"
    if ablation_suffix or metric_suffix or env_type != "crafter":
        print(f"[Log] CSV Path Adjusted: {summary_csv_path}")

    data_save_dir = Path(getattr(cfg.env.collect, "data_folder", str(TRAINER_PATH / "data")))
    
    # === A. 初始化 Temp Data Dir (阅后即焚隔离区) ===
    temp_data_dir = Path(
        getattr(
            cfg,
            "mac_temp_data_dir",
            str(TRAINER_PATH / "data" / env_type / "mini_tasks_temp" / "mac"),
        )
    )
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # [TARGET DATA] Domain-specific fixed target datasets
    target_data_dir = data_save_dir
    domain_cfg = cfg.domains[env_type] if hasattr(cfg, "domains") and env_type in cfg.domains else None
    if domain_cfg is not None and hasattr(domain_cfg, "target_tasks_folder"):
        target_data_dir = Path(domain_cfg.target_tasks_folder)
    # 清空之前的临时残留
    for f in os.listdir(temp_data_dir):
        if f.endswith(".npz"): 
            try: os.remove(temp_data_dir / f)
            except: pass

    # === 初始化 Summary CSV ===
    # Multi-seed runs should append into one shared CSV.
    file_exists = os.path.exists(summary_csv_path)
    file_non_empty = file_exists and os.path.getsize(summary_csv_path) > 0
    is_bipedal = (env_type == "bipedalwalker")
    is_minigrid = (env_type == "minigrid")
    if not file_non_empty:
        with open(summary_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            if is_bipedal:
                writer.writerow([
                    "Seed", "Iter",
                    "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
                    "gen_val_contact_acc", "gen_val_contact_bce", "gen_val_avg_val_loss_wm",
                    "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
                    "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len"
                ])
                print(f"[Logger] Experiment summary initialized with 16 columns at {summary_csv_path}")
            else:
                if is_minigrid:
                    csv_header = [
                        "Seed", "Iter",
                        "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
                        "gen_val_avg_val_loss_wm", "target_val_avg_val_loss_wm",
                        "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len"
                    ]
                else:
                    csv_header = [
                        "Seed", "Iter",
                        "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
                        "gen_val_val_inv_loss", "gen_val_val_ce_loss", "gen_val_avg_val_loss_wm",
                        "target_val_val_inv_loss", "target_val_val_ce_loss", "target_val_avg_val_loss_wm",
                        "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len"
                    ]
                writer.writerow(csv_header)
                print(f"[Logger] Experiment summary initialized at {summary_csv_path}")
    else:
        print(f"[Logger] Reusing existing summary CSV: {summary_csv_path}")
    print(f"[Logger] Experiment summary will be saved to {summary_csv_path}")
    
    # === Domain-aware single-rollout transition cap for MAC/DR ===
    _apply_domain_collection_budget(cfg, env_type)

    # [NEW] Force Fresh Start (Delete existing checkpoint if requested)
    ckpt_path = cfg.attention_model.model_save_path
    if getattr(cfg, "force_fresh_start", False):
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"[Fresh Start] Deleted existing checkpoint: {ckpt_path}")
        else:
            print(f"[Fresh Start] No checkpoint found to delete at: {ckpt_path}")

    # === A. 初始化 World Model ===
    wm_instance = AttentionWorldModel(cfg.attention_model).to(device)

    # === B. 初始化 Generator Interface ===
    gen_interface = GeneratorInterface(
        world_model=wm_instance,
        device=device,
        cfg=cfg,
        agent_type=cfg.generator_agent.agent_type
    )

    # === C. 初始化 Fisher Replay Buffer ===
    fisher_buffer = FisherReplayBuffer(
        max_size=cfg.attention_model.fisher_buffer_size,
        contact_positive_ratio=float(getattr(cfg.domains[cfg.domain], "contact_positive_ratio", 0.5))
    )

    # === D. 训练状态变量 ===
    old_params, fisher = None, None
    
    if os.path.exists(ckpt_path):
        print(f"[System] Found existing checkpoint at {ckpt_path}. Loading for resume...")
        try:
            # Fix for PyTorch 2.6 security change compatibility
            ckpt = torch.load(ckpt_path, weights_only=False)
            if 'state_dict' in ckpt:
                wm_instance.load_state_dict(ckpt['state_dict'])
            else:
                wm_instance.load_state_dict(ckpt)
            old_params = wm_instance.save_old_params()
            print("[System] Model weights resumed successfully.")
        except Exception as e:
            print(f"[Warning] Failed to resume from checkpoint: {e}. Starting fresh.")
    else:
        print(f"[System] No existing checkpoint found at {ckpt_path}. Starting from scratch.")

    total_iterations = cfg.generator_agent.total_iterations
    warmup_iterations = _safe_int_cfg(
        getattr(cfg.generator_agent, "warmup_iterations", 0),
        default=0,
        name="generator_agent.warmup_iterations",
    )
    wm_train_frequency = cfg.generator_agent.wm_train_frequency  
    warmup_cleanup_done = False

    # === E. 验证集定义 (fixed target tasks) ===
    if domain_cfg is not None and hasattr(domain_cfg, "target_task_prefix"):
        task_prefix = str(domain_cfg.target_task_prefix)
        task_suffix = str(getattr(domain_cfg, "target_task_suffix", "_uniform.npz"))
        task_start = int(getattr(domain_cfg, "target_task_start_idx", 0))
        task_count = int(getattr(domain_cfg, "target_task_count", 0))
        target_tasks = [f"{task_prefix}{i}" for i in range(task_start, task_start + task_count)]
        target_files = [f"{task_name}{task_suffix}" for task_name in target_tasks]
        print(f"[Config] Target tasks folder: {target_data_dir}")
    else:
        target_tasks = []
        target_files = []
        print(f"[Config] No fixed target task list configured for env_type='{env_type}'. Target validation will be skipped.")

    print(
        f">>> Starting UED Adversarial Training for {total_iterations} iterations..."
    )

    # --------------------------------------
    # 2. 主循环 (The Loop)
    # --------------------------------------
    for iteration in range(total_iterations):
        print(
            f"\n=== Iteration {iteration + 1}/{total_iterations} ==="
        )

        # --------------------------------------------------------
        # Step 0: Transition Handling (Warmup -> Adversarial)
        # --------------------------------------------------------
        if (not warmup_cleanup_done) and warmup_iterations > 0 and iteration >= warmup_iterations:
             print(
                 f"[System] Warmup ({warmup_iterations} iters) ended at iter={iteration}. "
                 "Clearing runtime generator buffers for fresh adversarial exploration."
             )
             gen_interface.clear_runtime_buffers()
             warmup_cleanup_done = True

        # --------------------------------------------------------
        # Step 0.5: 定期重置 Diversity Archive (防止模式坍塌)
        # --------------------------------------------------------
        div_reset_interval = int(getattr(cfg.generator_agent, "div_reset_interval", 0))
        if (
            warmup_cleanup_done
            and div_reset_interval > 0
            and (iteration - warmup_iterations) > 0
            and (iteration - warmup_iterations) % div_reset_interval == 0
        ):
            if hasattr(gen_interface, "diversity") and hasattr(gen_interface.diversity, "archive"):
                gen_interface.diversity.archive.clear()
                print(
                    f"[Diversity] Archive reset at iteration {iteration + 1} "
                    f"(every {div_reset_interval} iters). Generator will re-explore novel terrain."
                )
             
        # --------------------------------------------------------
        # Step 1: Generator 运行 (生成 -> 探索 -> 收集)
        # --------------------------------------------------------
        print(
            "[Generator] Generating environments and collecting trajectories..."
        )

        # [MODIFIED] Added 6 metrics support (Now returns 9 items)
        step_res = gen_interface.step(old_params=old_params, iteration=iteration)
        gen_val_avg_val_loss_wm = step_res[2]
        gen_val_aux_metric = step_res[3]
        gen_val_val_inv_loss = step_res[4]
        gen_val_contact_bce = step_res[4] if is_bipedal else 0.0
        gen_div_score = step_res[5]
        valid_trajectories = step_res[6]
        gen_solvable_count = step_res[7]
        gen_avg_bfs = step_res[8]
        gen_avg_ep_len = step_res[9] if len(step_res) > 9 else 0.0
        num_valid_trajs = len(valid_trajectories)
        print(
            f"[Generator] Collected {num_valid_trajs} valid trajectories. Solvable: {gen_solvable_count} | Avg BFS: {gen_avg_bfs:.2f}"
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
            # [BUG FIX] Must include inv/inv_next in buffer! 
            # Without this, Fisher Buffer has no inventory data,
            # and replay-based WM training completely lacks inventory supervision.
            # This was causing UED to have worse INV Loss than DR.
            if new_batch.get('rew') is not None:
                buffer_input["rew"] = new_batch["rew"]
            if new_batch.get('done') is not None:
                buffer_input["done"] = new_batch["done"]
            if new_batch.get('inv') is not None:
                buffer_input["inv"] = new_batch["inv"]
            if new_batch.get('inv_next') is not None:
                buffer_input["inv_next"] = new_batch["inv_next"]
            # Moved buffer update to AFTER training (Step 4.5) to prevent double counting


        # --------------------------------------------------------
        # Step 3: 更新 Generator (PPO Update)
        # --------------------------------------------------------
        # [MODIFIED] User Request: Warmup is for Generator now.
        # Generator trains from the start (Iter 0).
        if True:
            # [MODIFIED] Now returns entropy
            gen_loss, gen_entropy, gen_mean_reward = gen_interface.update()

            if gen_loss is not None:
                print(
                    f"[Generator] Policy Updated. Loss: {gen_loss:.4f} | Entropy: {gen_entropy:.4f}"
                )
            else:
                print("[Generator] Policy Updated. Loss: NaN (Skipped due to instability)")
                gen_loss = 0.0 
                gen_entropy = 0.0

        # --------------------------------------------------------
        # Step 4: 更新 World Model (Adversarial Learning)
        # --------------------------------------------------------
        wm_final_loss = 0.0 # default if not trained
        
        # [MODIFIED] User requested warmup to purely train Generator PPO.
        # No WM training and No Data Accumulation during warmup_iterations.
        is_warmup_for_wm = (iteration < warmup_iterations)
        
        if (not is_warmup_for_wm) and (iteration % wm_train_frequency == 0) and (new_batch is not None):
            print("[World Model] Retraining on current + replay data...")
            
            # --- [New Filter Logic] ---
            # [MODIFIED] Do NOT filter for training! Train on FULL batch.
            # Filtering only happens during Buffer Update (Step 4.5).
            # if new_batch is not None:
            #     new_batch = filter_balanced_batch(
            #         new_batch, 
            #         fisher_buffer, 
            #         ratio=cfg.attention_model.current_sample_ratio, 
            #         elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio
            #     )
            
            # Save FULL batch to disk in Temp Folder
            current_data_path = None
            if new_batch is not None:
                current_data_path = temp_data_dir / f"ued_training_set_iter_{iteration}.npz"
                save_dict = {
                    'a': new_batch['obs'].cpu().numpy() if torch.is_tensor(new_batch['obs']) else new_batch['obs'],
                    'b': new_batch['obs_next'].cpu().numpy() if torch.is_tensor(new_batch['obs_next']) else new_batch['obs_next'],
                    'c': new_batch['act'].cpu().numpy() if torch.is_tensor(new_batch['act']) else new_batch['act'],
                    'f': new_batch['info']
                }
                if new_batch.get('rew') is not None:
                    save_dict['d'] = new_batch['rew'].cpu().numpy() if torch.is_tensor(new_batch['rew']) else new_batch['rew']
                if new_batch.get('done') is not None:
                    save_dict['e'] = new_batch['done'].cpu().numpy() if torch.is_tensor(new_batch['done']) else new_batch['done']
                if new_batch.get('inv') is not None:
                     save_dict['g'] = new_batch['inv'].cpu().numpy() if torch.is_tensor(new_batch['inv']) else new_batch['inv']
                if new_batch.get('inv_next') is not None:
                     save_dict['h'] = new_batch['inv_next'].cpu().numpy() if torch.is_tensor(new_batch['inv_next']) else new_batch['inv_next']
                
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
            # [FIX] Clear stale hooks to prevent ReferenceError: weakly-referenced object no longer exists
            if hasattr(wm_instance, "_state_dict_hooks"):
                wm_instance._state_dict_hooks.clear()
            if hasattr(wm_instance, "_parameters"):
                for p_name, p in wm_instance._parameters.items():
                    if p is not None and hasattr(p, "_hooks"):
                        p._hooks.clear()

            # [MODIFIED] Corrected unpacking after train_api update (now returns 3-tuple: result_dict, fisher, net)
            train_res, fisher, _ = AttentionWM_training.train_api(
                cfg,
                wm_instance, 
                old_params,
                fisher,
                replay_data=replay_data,
            )
            # Update old_params from the result dict for next iteration
            old_params = train_res.get("old_params")
            
            # Step 4.2: Delete Temp Data After Training (阅后即焚)
            if current_data_path and os.path.exists(current_data_path):
                try: os.remove(current_data_path)
                except: pass
            
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
        if buffer_input is not None and not is_warmup_for_wm:
             fisher_buffer.update_combined(
                buffer_input,
                cfg.attention_model.current_sample_ratio,
                cfg.attention_model.fisher_buffer_elements_ratio,
            )
             print(
                f"[Buffer] Archived {new_data_size} transitions. "
                f"Buffer Size: {len(fisher_buffer)}"
            )
        elif is_warmup_for_wm:
             print("[Buffer] Warmup phase: Skipping data accumulation to match budget.")
        
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
        # [MODIFIED] Validation Logic:
        # 1. Warmup: Skip validation to save time.
        # 2. Training: Validate every step to track progress.
        warmup_iters = _safe_int_cfg(
            getattr(cfg.generator_agent, "warmup_iterations", 0),
            default=0,
            name="generator_agent.warmup_iterations",
        )
        if target_files and iteration >= (warmup_iters - 1): 
            print(f"\n>>> Validating on Target Tasks...")
            target_ce_losses = []
            target_inv_losses = []
            target_avg_losses = []
            target_contact_accs = []
            target_contact_bces = []
            
            # Temporary set to validation mode
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = True

            # Fix: Disable WandB during validation to prevent Hook ReferenceError and run spam
            old_use_wandb = cfg.attention_model.use_wandb
            cfg.attention_model.use_wandb = False

            for t_name, t_file in zip(target_tasks, target_files):
                full_target_path = os.path.join(str(target_data_dir), t_file)
                if not os.path.exists(full_target_path):
                    continue
                res_dict = validate_on_target_task(
                    cfg, 
                    net=wm_instance, 
                    old_params=None, 
                    data_save_dir=str(target_data_dir), 
                    target_file=t_file, 
                    phase_name=f"Iter_{iteration}",
                    VALID_TIMES=1
                )
                
                if res_dict:
                    target_avg_losses.append(res_dict['avg_val_loss_wm'])
                    if is_bipedal:
                        target_contact_accs.append(res_dict.get('contact_acc', 0.0))
                        target_contact_bces.append(res_dict.get('contact_bce', 0.0))
                    elif not is_minigrid:
                        target_ce_losses.append(res_dict.get('terrain_loss', 0.0))
                        target_inv_losses.append(res_dict.get('inventory_loss', 0.0))
            
            # Restore configs
            cfg.attention_model.freeze_weight = old_freeze
            cfg.attention_model.use_wandb = old_use_wandb

            # Aggregate Targets
            if target_avg_losses:
                target_val_avg_val_loss_wm = float(np.mean(target_avg_losses))
                if is_bipedal:
                    target_val_contact_acc = float(np.mean(target_contact_accs)) if target_contact_accs else 0.0
                    target_val_contact_bce = float(np.mean(target_contact_bces)) if target_contact_bces else 0.0
                    print(f"[Metrics] Combined Target Loss -> Total: {target_val_avg_val_loss_wm:.4f} | Contact Acc: {target_val_contact_acc:.4f} | Contact BCE: {target_val_contact_bce:.4f}")
                elif not is_minigrid:
                    target_val_val_ce_loss = float(np.mean(target_ce_losses))
                    target_val_val_inv_loss = float(np.mean(target_inv_losses))
                    print(f"[Metrics] Combined Target Loss -> Total: {target_val_avg_val_loss_wm:.4f} | Terrain: {target_val_val_ce_loss:.4f}")
                else:
                    print(f"[Metrics] Combined Target Loss -> Total: {target_val_avg_val_loss_wm:.4f}")
            else:
                target_val_avg_val_loss_wm = 0.0
                if is_bipedal:
                    target_val_contact_acc = 0.0
                    target_val_contact_bce = 0.0
                elif not is_minigrid:
                    target_val_val_ce_loss = 0.0
                    target_val_val_inv_loss = 0.0
        else:
            target_val_avg_val_loss_wm = 0.0
            if is_bipedal:
                target_val_contact_acc = 0.0
                target_val_contact_bce = 0.0
            elif not is_minigrid:
                target_val_val_ce_loss = 0.0
                target_val_val_inv_loss = 0.0

        # --------------------------------------------------------
        # Step 6: 写 CSV 日志 (Experiment Summary)
        # --------------------------------------------------------
        if summary_csv_path is not None:
            try:
                with open(summary_csv_path, mode='a', newline='') as f:
                    # --- [Symmetrical 6-Column Metrics] ---
                    # Ensure we don't log NaN if lists are empty
                    gen_div_reward_val = gen_div_score if gen_div_score is not None else 0.0

                    # Prepare row as a dictionary (Exact Column Order Alignment)
                    if is_bipedal:
                        row_data = {
                            "Seed": seed,
                            "Iter": iteration + 1,
                            "Gen_Mean_Reward": f"{gen_mean_reward:.4f}",
                            "Gen_Loss": f"{gen_loss:.4f}",
                            "Gen_Entropy": f"{gen_entropy:.4f}",
                            "Gen_Div_Reward": f"{gen_div_reward_val:.4f}",
                            "gen_val_contact_acc": f"{gen_val_aux_metric:.6f}",
                            "gen_val_contact_bce": f"{gen_val_contact_bce:.6f}",
                            "gen_val_avg_val_loss_wm": f"{gen_val_avg_val_loss_wm:.6f}",
                            "target_val_contact_acc": f"{target_val_contact_acc:.6f}",
                            "target_val_contact_bce": f"{target_val_contact_bce:.6f}",
                            "target_val_avg_val_loss_wm": f"{target_val_avg_val_loss_wm:.6f}",
                            "New_Data_Size": new_data_size,
                            "Buffer_Size": len(fisher_buffer),
                            "Solvable_Count": f"{gen_solvable_count}",
                            "Avg_Path_Len": f"{gen_avg_ep_len:.2f}"
                        }
                    else:
                        if is_minigrid:
                            row_data = {
                                "Seed": seed,
                                "Iter": iteration + 1,
                                "Gen_Mean_Reward": f"{gen_mean_reward:.4f}",
                                "Gen_Loss": f"{gen_loss:.4f}",
                                "Gen_Entropy": f"{gen_entropy:.4f}",
                                "Gen_Div_Reward": f"{gen_div_reward_val:.4f}",
                                "gen_val_avg_val_loss_wm": f"{gen_val_avg_val_loss_wm:.6f}",
                                "target_val_avg_val_loss_wm": f"{target_val_avg_val_loss_wm:.6f}",
                                "New_Data_Size": new_data_size,
                                "Buffer_Size": len(fisher_buffer),
                                "Solvable_Count": f"{gen_solvable_count}",
                                "Avg_Path_Len": f"{gen_avg_bfs:.2f}"
                            }
                        else:
                            row_data = {
                                "Seed": seed,
                                "Iter": iteration + 1,
                                "Gen_Mean_Reward": f"{gen_mean_reward:.4f}",
                                "Gen_Loss": f"{gen_loss:.4f}",
                                "Gen_Entropy": f"{gen_entropy:.4f}",
                                "Gen_Div_Reward": f"{gen_div_reward_val:.4f}",
                                "gen_val_val_inv_loss": f"{gen_val_val_inv_loss:.6f}",
                                "gen_val_val_ce_loss": f"{gen_val_aux_metric:.6f}",
                                "gen_val_avg_val_loss_wm": f"{gen_val_avg_val_loss_wm:.6f}",
                                "target_val_val_inv_loss": f"{target_val_val_inv_loss:.6f}",
                                "target_val_val_ce_loss": f"{target_val_val_ce_loss:.6f}",
                                "target_val_avg_val_loss_wm": f"{target_val_avg_val_loss_wm:.6f}",
                                "New_Data_Size": new_data_size,
                                "Buffer_Size": len(fisher_buffer),
                                "Solvable_Count": f"{gen_solvable_count}",
                                "Avg_Path_Len": f"{gen_avg_ep_len:.2f}"
                            }

                    writer = csv.writer(f)
                    writer.writerow(list(row_data.values()))

            except Exception as e:
                print(f"[Error] Failed to write CSV log: {e}")

        # --------------------------------------------------------
        # Step 7: Cleanup Temporary Data
        # --------------------------------------------------------
        # Delete generated trajectory files for this iteration to save space
        # Pattern matches UED_Dual_iter{iteration}_b{idx}_test_{explore_type}.npz
        temp_files = glob.glob(str(data_save_dir / f"UED_Dual_iter{iteration}_b*.npz"))
        if temp_files:
            print(f"[Cleanup] Deleting {len(temp_files)} temporary files for Iteration {iteration}...")
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"[Warning] Could not delete {f}: {e}")
            print(f"[Cleanup] Done.")

    print(">>> UED Adversarial Training Finished.")


@hydra.main(
    version_base=None,
    config_path=str(TRAINER_PATH / "conf"),
    config_name="config_mac",
)
def adversarial_ued_training_wrapper(cfg: DictConfig):
    """Wrapper for running a single seed"""
    adversarial_ued_training(cfg)


if __name__ == "__main__":
    # 默认直接运行，Hydra 会自动从 config_UED.yaml 读取配置（包括 ablation 和 seed）。
    # 如果需要运行多个 seed 或 ablation，可以使用 Hydra 的 multirun 功能：
    # python UED_wm_learning.py -m seed=0,1,2 ablation.type=none,no_diversity
    adversarial_ued_training_wrapper()
