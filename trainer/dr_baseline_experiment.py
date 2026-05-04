import os
import sys
import tempfile
import torch
import numpy as np
import pandas as pd
import math
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
    set_seed, validate_on_target_task, validate_on_all_targets, convert_trajectories_to_batch
)


def _safe_int_cfg(value, default=0, name="value"):
    if value is None:
        print(f"[Config Warning] {name} is None, fallback to {default}.")
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"[Config Warning] {name}={value} is invalid, fallback to {default}.")
        return int(default)


def _apply_domain_collection_budget(cfg: DictConfig, domain_name: str):
    """
    Configure per-rollout collection size from domain-level per-iteration budget.
    effective_per_rollout = ceil(iter_transition_budget / generator_batch_size)
    """
    if not hasattr(cfg, "domains") or domain_name not in cfg.domains:
        print(f"[Config Warning] Missing domains.{domain_name}; keep existing collection settings.")
        return

    domain_cfg = cfg.domains[domain_name]
    with open_dict(cfg):
        batch_override = getattr(domain_cfg, "generator_batch_size", None)
        if batch_override is not None:
            cfg.generator_agent.batch_size = _safe_int_cfg(
                batch_override,
                default=getattr(cfg.generator_agent, "batch_size", 8),
                name=f"domains.{domain_name}.generator_batch_size",
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
        name=f"domains.{domain_name}.iter_transition_budget",
    )
    per_rollout_max = max(1, int(math.ceil(iter_budget / batch_size)))

    with open_dict(cfg):
        cfg.env.collect.maximum_dataset_size = per_rollout_max
        cfg.env.collect.mini_dataset_size = per_rollout_max

    expected_total = per_rollout_max * batch_size
    print(
        f"[Config] Domain budget applied | domain={domain_name} | "
        f"iter_budget={iter_budget} | batch_size={batch_size} | "
        f"per_rollout_max={per_rollout_max} | expected_iter_total={expected_total}"
    )


def _ensure_csv_header_compatible(csv_path: Path, expected_columns):
    """
    If an existing CSV has a different header, back it up to avoid mixed-schema append.
    Returns whether the target csv_path should be treated as existing for append.
    """
    if (not csv_path.exists()) or csv_path.stat().st_size == 0:
        return False

    expected_header = ",".join(expected_columns)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            current_header = f.readline().strip()
    except Exception:
        current_header = ""

    if current_header == expected_header:
        return True

    backup_path = Path(f"{csv_path}.legacy_backup")
    suffix_idx = 1
    while backup_path.exists():
        backup_path = Path(f"{csv_path}.legacy_backup{suffix_idx}")
        suffix_idx += 1

    os.replace(csv_path, backup_path)
    print(
        f"[Logger] Existing CSV header mismatch. Backed up old file to {backup_path} "
        f"and starting a new summary file."
    )
    return False

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
    is_bipedal = (domain_name == "bipedalwalker")
    is_minigrid = (domain_name == "minigrid")
    val_n_phases = int(getattr(d_cfg, "val_n_phases", getattr(d_cfg, "target_task_count", 20)))
    val_task_prefix = str(getattr(d_cfg, "val_task_prefix", getattr(d_cfg, "target_task_prefix", "target_task")))
    val_data_path = str(getattr(d_cfg, "val_data_path", getattr(d_cfg, "target_tasks_folder", "")))
    val_suffix = str(getattr(d_cfg, "val_suffix", getattr(d_cfg, "target_task_suffix", "_uniform.npz")))
    val_start_idx = int(getattr(d_cfg, "start_idx", getattr(d_cfg, "target_task_start_idx", 0)))
    
    print(f"\n{'='*80}")
    print(f"### [DR BASEMENT START] Domain: {domain_name.upper()} | Seed: {seed}")
    print(f"{'='*80}\n")

    # Override Attention Model configs based on Domain
    with open_dict(cfg):
        cfg.attention_model.env_type = domain_name
        cfg.attention_model.grid_shape = d_cfg.grid_shape
        cfg.attention_model.obs_norm_values = d_cfg.obs_norm
        cfg.attention_model.action_norm_values = d_cfg.action_norm
        cfg.attention_model.validation_metric = d_cfg.validation_metric
        cfg.attention_model.data_type = d_cfg.data_type
    _apply_domain_collection_budget(cfg, domain_name)

    # [NEW] Force Fresh Start (Delete existing checkpoint if requested)
    ckpt_path = cfg.attention_model.model_save_path
    force_fresh_start = bool(getattr(cfg, "force_fresh_start", False))
    if force_fresh_start:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"[Fresh Start] Deleted existing checkpoint: {ckpt_path}")
        else:
            print(f"[Fresh Start] No checkpoint found to delete at: {ckpt_path}")

    # 1. Initialization
    wm = AttentionWorldModel(cfg.attention_model).to(device)

    # DR usually generates random maps via GeneratorInterface (agent_type='random')
    generator = GeneratorInterface(wm, device, cfg, agent_type='random')
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size, contact_positive_ratio=float(getattr(cfg.domains[cfg.domain], "contact_positive_ratio", 0.5))
)
    
    log_dir = Path(cfg.dr_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    temp_data_dir = Path(cfg.dr_temp_data_dir)
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 强制将所有 UED 收集到的数据存在 dr temp 文件夹中
    cfg.env.collect.data_folder = str(temp_data_dir) + "/"

    mask_suffix = f"_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"
    summary_csv_path = log_dir / f"dr_summary_{domain_name}{mask_suffix}.csv"
    file_exists = False

    old_params, fisher = None, None
    if is_minigrid:
        csv_columns = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_avg_val_loss_wm", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
        ]
    elif not is_bipedal: # Crafter or others
        csv_columns = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_val_inv_loss", "gen_val_val_ce_loss", "gen_val_avg_val_loss_wm",
            "target_val_val_inv_loss", "target_val_val_ce_loss", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
        ]
    elif is_bipedal:
        csv_columns = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_contact_acc", "gen_val_contact_bce", "gen_val_avg_val_loss_wm",
            "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
        ]
    file_exists = _ensure_csv_header_compatible(summary_csv_path, csv_columns)

    # 1.5 Load existing model if available (Resume logic)
    ckpt_path = cfg.attention_model.model_save_path
    if force_fresh_start:
        print("[System] Fresh-start mode enabled. Skipping checkpoint resume.")
    else:
        if os.path.exists(ckpt_path):
            print(f"[System] Found existing checkpoint at {ckpt_path}. Loading weights for resume...")
            try:
                ckpt = torch.load(ckpt_path, weights_only=False)
                if 'state_dict' in ckpt:
                    wm.load_state_dict(ckpt['state_dict'])
                else:
                    wm.load_state_dict(ckpt)
                old_params = wm.save_old_params()
                print("[System] Model weights loaded successfully.")
            except Exception as e:
                print(f"[Warning] Failed to load existing model: {e}. Starting from scratch.")
        else:
            print(f"[System] No existing checkpoint found at {ckpt_path}. Starting from scratch.")

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
            aux_metric = float(trajs[3])
            inv_loss_or_bce = float(trajs[4])
            if is_bipedal:
                gen_val_contact_acc = aux_metric
                gen_val_contact_bce = inv_loss_or_bce
                gen_val_val_ce_loss = 0.0
                gen_val_val_inv_loss = 0.0
            else:
                gen_val_val_ce_loss = aux_metric
                gen_val_val_inv_loss = inv_loss_or_bce
                gen_val_contact_acc = 0.0
                gen_val_contact_bce = 0.0
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
            gen_val_contact_acc = 0.0
            gen_val_contact_bce = 0.0
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
        # [ALIGN] Respect warmup: freeze WM training during warmup (same as MAC)
        warmup_iters_wm = _safe_int_cfg(
            getattr(cfg.generator_agent, "warmup_iterations", 0),
            default=0,
            name="generator_agent.warmup_iterations",
        )
        is_warmup = (iteration < warmup_iters_wm)
        if (not is_warmup) and (iteration % cfg.generator_agent.wm_train_frequency == 0):
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
                    'a': new_batch['obs'].cpu().numpy() if torch.is_tensor(new_batch['obs']) else new_batch['obs'],
                    'b': new_batch['obs_next'].cpu().numpy() if torch.is_tensor(new_batch['obs_next']) else new_batch['obs_next'],
                    'c': new_batch['act'].cpu().numpy() if torch.is_tensor(new_batch['act']) else new_batch['act'],
                }
                if new_batch.get('rew') is not None:
                    save_dict['d'] = new_batch['rew'].cpu().numpy() if torch.is_tensor(new_batch['rew']) else new_batch['rew']
                if new_batch.get('done') is not None:
                    save_dict['e'] = new_batch['done'].cpu().numpy() if torch.is_tensor(new_batch['done']) else new_batch['done']
                if new_batch.get('info') is not None:
                    save_dict['f'] = new_batch['info']
                if new_batch.get('inv') is not None:
                    save_dict['g'] = new_batch['inv'].cpu().numpy() if torch.is_tensor(new_batch['inv']) else new_batch['inv']
                if new_batch.get('inv_next') is not None:
                    save_dict['h'] = new_batch['inv_next'].cpu().numpy() if torch.is_tensor(new_batch['inv_next']) else new_batch['inv_next']

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
        warmup_iters = _safe_int_cfg(
            getattr(cfg.generator_agent, "warmup_iterations", 0),
            default=0,
            name="generator_agent.warmup_iterations",
        )
        if iteration >= (warmup_iters - 1):
            print(f"  [Validation] Running zero-shot test on all {val_n_phases} targets...")
            task_indices = range(val_start_idx, val_start_idx + val_n_phases)
            task_names = [f"{val_task_prefix}{v_idx}" for v_idx in task_indices]
            val_summary = validate_on_all_targets(
                cfg,
                wm,
                val_data_path,
                task_names,
                val_suffix,
                phase_name=f"dr_iter_{iteration+1}",
                VALID_TIMES=1,
            )

            if val_summary["valid_count"] > 0:
                target_val_avg_val_loss_wm = val_summary["avg_val_loss_wm"]
                if is_bipedal:
                    target_val_contact_acc = val_summary.get("contact_acc", 0.0)
                    target_val_contact_bce = val_summary.get("contact_bce", 0.0)
                    target_val_val_ce_loss = 0.0
                    target_val_val_inv_loss = 0.0
                    print(f"    -> Results: Avg Loss = {target_val_avg_val_loss_wm:.5f}")
                else:
                    target_val_val_ce_loss = val_summary.get("terrain_loss", 0.0)
                    target_val_val_inv_loss = val_summary.get("inventory_loss", 0.0)
                    target_val_contact_acc = 0.0
                    target_val_contact_bce = 0.0
                    print(f"    -> Results: Avg Loss = {target_val_avg_val_loss_wm:.5f}")
            else:
                target_val_avg_val_loss_wm = 0.0
                target_val_val_ce_loss = 0.0
                target_val_val_inv_loss = 0.0
                target_val_contact_acc = 0.0
                target_val_contact_bce = 0.0
        else:
            target_val_avg_val_loss_wm = 0.0
            target_val_val_ce_loss = 0.0
            target_val_val_inv_loss = 0.0
            target_val_contact_acc = 0.0
            target_val_contact_bce = 0.0

        # D. Buffer Archiving
        fisher_buffer.add_from_batch(
            new_batch, 
            current_sample_ratio=cfg.attention_model.current_sample_ratio,
            fisher_buffer_elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio
        )
        print(f"  [Buffer] Archived {current_transitions} transitions. Buffer Size: {len(fisher_buffer)}")
        if is_minigrid:
            row_data = {
                "Seed": seed, "Iter": iteration + 1, "Gen_Mean_Reward": 0.0, "Gen_Loss": 0.0,
                "Gen_Entropy": 0.0, "Gen_Div_Reward": gen_div_reward,
                "gen_val_avg_val_loss_wm": gen_val_avg_val_loss_wm,
                "target_val_avg_val_loss_wm": target_val_avg_val_loss_wm,
                "New_Data_Size": current_transitions, "Buffer_Size": len(fisher_buffer),
                "Solvable_Count": solvable_count, "Avg_Path_Len": avg_path_len,
            }
        elif is_bipedal:
            row_data = {
                "Seed": seed, "Iter": iteration + 1, "Gen_Mean_Reward": 0.0, "Gen_Loss": 0.0,
                "Gen_Entropy": 0.0, "Gen_Div_Reward": gen_div_reward,
                "gen_val_contact_acc": gen_val_contact_acc, "gen_val_contact_bce": gen_val_contact_bce,
                "gen_val_avg_val_loss_wm": gen_val_avg_val_loss_wm,
                "target_val_contact_acc": target_val_contact_acc, "target_val_contact_bce": target_val_contact_bce,
                "target_val_avg_val_loss_wm": target_val_avg_val_loss_wm,
                "New_Data_Size": current_transitions, "Buffer_Size": len(fisher_buffer),
                "Solvable_Count": solvable_count, "Avg_Path_Len": avg_path_len,
            }
        else:
            row_data = {
                "Seed": seed, "Iter": iteration + 1, "Gen_Mean_Reward": 0.0, "Gen_Loss": 0.0,
                "Gen_Entropy": 0.0, "Gen_Div_Reward": gen_div_reward,
                "gen_val_val_inv_loss": gen_val_val_inv_loss, "gen_val_val_ce_loss": gen_val_val_ce_loss,
                "gen_val_avg_val_loss_wm": gen_val_avg_val_loss_wm,
                "target_val_val_inv_loss": target_val_val_inv_loss, "target_val_val_ce_loss": target_val_val_ce_loss,
                "target_val_avg_val_loss_wm": target_val_avg_val_loss_wm,
                "New_Data_Size": current_transitions, "Buffer_Size": len(fisher_buffer),
                "Solvable_Count": solvable_count, "Avg_Path_Len": avg_path_len,
            }

        pd.DataFrame([row_data], columns=csv_columns).to_csv(
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
