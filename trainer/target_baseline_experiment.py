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
from trainer.common.utils import collect_data_general
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.common.utils import TRAINER_PATH
from trainer.common.utils import set_seed

BASELINE_COLUMNS = [
    "Seed",
    "Iter",
    "Phase",
    "Trained_On",
    "data_size",
    "cumulative_data_size",
    "target_val_val_ce_loss",
    "target_val_val_inv_loss",
    "target_val_avg_val_loss_wm",
    "Avg_Val_CE",
    "Avg_Val_INV",
    "Avg_Val_Total",
]


def _ensure_baseline_csv(csv_out_path: Path) -> bool:
    """
    Ensure output CSV has a valid header schema.
    Returns True if file exists with valid header, else False.
    """
    if not csv_out_path.exists():
        return False

    with open(csv_out_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        return False

    first_col = first_line.split(",")[0].strip() if "," in first_line else first_line
    if first_col in BASELINE_COLUMNS:
        return True

    # Legacy no-header file. Try to recover and rewrite with header.
    legacy_backup = csv_out_path.with_name(f"{csv_out_path.stem}_no_header_backup{csv_out_path.suffix}")
    legacy_df = pd.read_csv(csv_out_path, header=None)
    legacy_df.to_csv(legacy_backup, index=False, header=False)

    if legacy_df.shape[1] == len(BASELINE_COLUMNS):
        legacy_df.columns = BASELINE_COLUMNS
        legacy_df.to_csv(csv_out_path, index=False)
        print(f"[Repair] Added header for existing CSV. Backup: {legacy_backup}")
        return True

    # Legacy 7-col rows seen in current workflow:
    # Seed,Iter,Phase,Trained_On,target_val_val_ce_loss,target_val_val_inv_loss,target_val_avg_val_loss_wm
    if legacy_df.shape[1] == 7:
        legacy_df.columns = [
            "Seed",
            "Iter",
            "Phase",
            "Trained_On",
            "target_val_val_ce_loss",
            "target_val_val_inv_loss",
            "target_val_avg_val_loss_wm",
        ]
        legacy_df["data_size"] = np.nan
        legacy_df["cumulative_data_size"] = np.nan
        legacy_df["Avg_Val_CE"] = legacy_df["target_val_val_ce_loss"]
        legacy_df["Avg_Val_INV"] = legacy_df["target_val_val_inv_loss"]
        legacy_df["Avg_Val_Total"] = legacy_df["target_val_avg_val_loss_wm"]
        legacy_df = legacy_df[BASELINE_COLUMNS]
        legacy_df.to_csv(csv_out_path, index=False)
        print(f"[Repair] Upgraded legacy 7-column CSV with header. Backup: {legacy_backup}")
        return True

    # Unknown schema: preserve old file and start a fresh CSV.
    broken_path = csv_out_path.with_name(f"{csv_out_path.stem}_broken_schema{csv_out_path.suffix}")
    os.replace(csv_out_path, broken_path)
    print(
        f"[Repair] Unexpected CSV schema ({legacy_df.shape[1]} cols). "
        f"Moved old CSV to {broken_path}. Backup: {legacy_backup}"
    )
    return False


@hydra.main(version_base=None, config_path="conf", config_name="config_cl")
def run_target_baseline_experiment(cfg: DictConfig):
    """
    Unified Baseline Script: Trains World Model directly on Target Tasks data.
    Supports both Crafter (10 tasks) and MiniGrid (20 tasks).
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
    cumulative_data_size = 0
    
    log_dir = TRAINER_PATH / "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_out_path = log_dir / "crafter_baseline_results.csv"
    file_exists = _ensure_baseline_csv(csv_out_path)

    # 3. Main Phase Loop (Iterating through tasks)
    n_phases = d_cfg.n_phases
    task_indices = range(d_cfg.start_idx, d_cfg.start_idx + n_phases)
    task_names = [f"{d_cfg.task_prefix}{i}" for i in task_indices]
    target_dataset_size = 20000
    force_recollect_per_task = True

    for i, current_task in enumerate(task_names):
        # Force a unique dataset for each random seed to evaluate robustness fairly
        train_path = os.path.join(d_cfg.data_path, f"{current_task}_seed{seed}{d_cfg.train_suffix}")
        
        print(f"\n[PHASE {i+1}/{n_phases}] Training on: {current_task} (Seed: {seed})")
        if force_recollect_per_task or (not os.path.exists(train_path)):
            print(
                f"Collecting {target_dataset_size} interactions for {current_task} "
                f"(Seed: {seed})..."
            )
            from omegaconf import OmegaConf
            
            col_cfg = copy.deepcopy(cfg)
            if not hasattr(col_cfg, "env"):
                with open_dict(col_cfg):
                    col_cfg.env = OmegaConf.create({})
            with open_dict(col_cfg.env):
                col_cfg.env.collect = OmegaConf.create({
                    "episodes": 1000000, 
                    "maximum_dataset_size": target_dataset_size, 
                    "data_type": "random",
                    "data_folder": str(d_cfg.data_path),
                    "data_save_path": "",
                    "visualize_save_path": "",
                    "visualize_filename": "",
                    "save_coverage_visualize": False,
                    "save_env_visualize": False
                })
            
            env_txt = os.path.join(str(TRAINER_PATH), "level", domain_name, "target_tasks", f"{current_task}.txt")
            
            # Run collection with config lock completely opened
            with open_dict(col_cfg.env.collect):
                saved_file = collect_data_general(
                    col_cfg,
                    env_source=env_txt,
                    save_name=f"{current_task}_seed{seed}",
                    max_steps=5000,
                    maximum_dataset_size=target_dataset_size,
                    recollect_data=True
                )
            
            # ensure filename matches what we need
            if saved_file != train_path and os.path.exists(saved_file):
                import shutil
                shutil.move(saved_file, train_path)
                print(f"Moved {saved_file} to {train_path}")
                
            if not os.path.exists(train_path):
                print(f"Failed to generate {train_path}. Skipping phase.")
                continue

        # Load data and slice for granular training (optional, following crafter_baseline logic)
        full_data = np.load(train_path, allow_pickle=True)
        data_keys = list(full_data.keys())
        total_len = len(full_data[data_keys[0]])
        
        # Enforce exact 20,000 data transition limit (uniformly sampled/truncated)
        limit_size = target_dataset_size
        if total_len > limit_size:
            print(f"  -> Truncating {total_len} to {limit_size} exactly as requested.")
            full_data = {k: full_data[k][:limit_size] for k in data_keys}
            total_len = limit_size
        
        # One task = one phase: use all available (targeted 20,000) data in a single update.
        sub_step_size = total_len
        n_sub_steps = 1
        
        for sub_idx in range(n_sub_steps):
            print(f"  -> Single-Step update (Iter: {i+1})")
            
            start_i = sub_idx * sub_step_size
            end_i = min((sub_idx + 1) * sub_step_size, total_len)
            sub_data = {k: full_data[k][start_i:end_i] for k in data_keys}
            current_data_size = end_i - start_i
            cumulative_data_size += current_data_size

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
                # Prepare metrics dict for this sub-step
                phase_metrics = {
                    "Seed": seed,
                    "Iter": f"{i+1}",
                    "Phase": f"{i+1}",
                    "Trained_On": current_task,
                    "data_size": current_data_size,
                    "cumulative_data_size": cumulative_data_size,
                }
                
                sum_total = 0.0
                sum_ce = 0.0
                sum_inv = 0.0
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
                    ce_val = float(metrics.get('val/ce_loss', metrics.get('terrain_loss', 0.0)))
                    inv_val = float(metrics.get('val/inv_loss', metrics.get('inventory_loss', 0.0)))
                    
                    sum_total += l_val
                    sum_ce += ce_val
                    sum_inv += inv_val
                    valid_count += 1
                    
                if valid_count > 0:
                    # keep both naming styles for downstream compatibility
                    phase_metrics["target_val_val_ce_loss"] = sum_ce / valid_count
                    phase_metrics["target_val_val_inv_loss"] = sum_inv / valid_count
                    phase_metrics["target_val_avg_val_loss_wm"] = sum_total / valid_count

                    phase_metrics["Avg_Val_CE"] = sum_ce / valid_count
                    phase_metrics["Avg_Val_INV"] = sum_inv / valid_count
                    phase_metrics["Avg_Val_Total"] = sum_total / valid_count
                    print(f"    -> Results: Avg CE = {phase_metrics['Avg_Val_CE']:.5f}, Avg INV = {phase_metrics['Avg_Val_INV']:.5f}, Avg Total = {phase_metrics['Avg_Val_Total']:.5f}")
                
                # Append this sub-step's metrics to CSV
                pd.DataFrame([phase_metrics], columns=BASELINE_COLUMNS).to_csv(
                    csv_out_path,
                    mode='a',
                    header=not file_exists,
                    index=False,
                )
                file_exists = True

        # 4. Finalize Phase: Archive to Replay Buffer
        print(f"  [Buffer] Archiving {current_task} data...")
        fisher_buffer.add_from_npz(train_path, current_sample_ratio=cfg.attention_model.ewc_ratio)
        torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Experiment Complete. CSV saved to: {csv_out_path}")

if __name__ == "__main__":
    run_target_baseline_experiment()
