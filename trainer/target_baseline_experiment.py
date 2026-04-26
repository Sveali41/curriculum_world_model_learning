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
from trainer.common.utils import set_seed, validate_on_all_targets



def _resolve_level_txt_path(level_folder: str, task_name: str) -> str:
    """
    Resolve level txt path across historical folder naming variants.
    MiniGrid used `target_task/` in this repo, while other domains use `target_tasks/`.
    """
    level_root = Path(TRAINER_PATH) / "level" / level_folder
    candidates = [
        level_root / "target_tasks" / f"{task_name}.txt",
        level_root / "target_task" / f"{task_name}.txt",
        level_root / f"{task_name}.txt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Could not find level txt for task '{task_name}'. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _extract_map_hw(obs_arr) -> tuple[int, int] | None:
    """
    Infer (H, W) from image-like observation arrays.
    Supports NHWC/NCHW with either batch or single sample.
    """
    if not isinstance(obs_arr, np.ndarray):
        return None

    # Batched image tensor
    if obs_arr.ndim == 4:
        # NHWC: (B, H, W, C)
        if obs_arr.shape[-1] <= 4 and obs_arr.shape[1] > 4:
            return int(obs_arr.shape[1]), int(obs_arr.shape[2])
        # NCHW: (B, C, H, W)
        if obs_arr.shape[1] <= 4 and obs_arr.shape[-1] > 4:
            return int(obs_arr.shape[2]), int(obs_arr.shape[3])
        # Fallback
        return int(obs_arr.shape[1]), int(obs_arr.shape[2])

    # Single image tensor
    if obs_arr.ndim == 3:
        # HWC
        if obs_arr.shape[-1] <= 4 and obs_arr.shape[0] > 4:
            return int(obs_arr.shape[0]), int(obs_arr.shape[1])
        # CHW
        if obs_arr.shape[0] <= 4 and obs_arr.shape[-1] > 4:
            return int(obs_arr.shape[1]), int(obs_arr.shape[2])
        # Fallback
        return int(obs_arr.shape[0]), int(obs_arr.shape[1])

    return None


def _ensure_baseline_csv(csv_out_path: Path, csv_columns: list) -> bool:
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
    if first_col in csv_columns:
        return True

    # Legacy no-header file. Try to recover and rewrite with header.
    legacy_backup = csv_out_path.with_name(f"{csv_out_path.stem}_no_header_backup{csv_out_path.suffix}")
    legacy_df = pd.read_csv(csv_out_path, header=None)
    legacy_df.to_csv(legacy_backup, index=False, header=False)

    if legacy_df.shape[1] == len(csv_columns):
        legacy_df.columns = csv_columns
        legacy_df.to_csv(csv_out_path, index=False)
        print(f"[Repair] Added header for existing CSV. Backup: {legacy_backup}")
        return True

    # Legacy 7-col rows seen in current workflow:
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
        if "target_val_contact_acc" in csv_columns:
            legacy_df["target_val_contact_acc"] = np.nan
            legacy_df["target_val_contact_bce"] = np.nan
        if "Avg_Val_CE" in csv_columns:
            legacy_df["Avg_Val_CE"] = legacy_df["target_val_val_ce_loss"]
            legacy_df["Avg_Val_INV"] = legacy_df["target_val_val_inv_loss"]
        legacy_df["Avg_Val_Total"] = legacy_df["target_val_avg_val_loss_wm"]
        
        # Keep only required columns filling empty ones with Nan
        for col in csv_columns:
            if col not in legacy_df.columns:
                legacy_df[col] = np.nan
        legacy_df = legacy_df[csv_columns]
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


@hydra.main(version_base=None, config_path="conf", config_name="config_target_baseline")
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
        cfg.attention_model.validation_metric = d_cfg.validation_metric
        cfg.attention_model.data_type = d_cfg.data_type

    # 2. Initialization
    net = AttentionWorldModel(cfg.attention_model).to(device)
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    
    old_params = None
    fisher = None
    cumulative_data_size = 0
    fisher_target_shape = None  # (H, W) for MiniGrid replay shape harmonization
    
    log_dir = TRAINER_PATH / "logs" / "results_target_baseline"
    os.makedirs(log_dir, exist_ok=True)
    
    is_bipedal = (domain_name == "bipedalwalker")
    is_crafter = (domain_name == "crafter")
    
    if is_bipedal:
        csv_columns = [
            "Seed", "Iter", "Phase", "Trained_On", "data_size", "cumulative_data_size",
            "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
            "Avg_Val_Total",
        ]
        csv_out_path = log_dir / "target_baseline_bipedalwalker.csv"
    else:
        csv_columns = [
            "Seed", "Iter", "Phase", "Trained_On", "data_size", "cumulative_data_size",
            "target_val_val_ce_loss", "target_val_val_inv_loss", "target_val_avg_val_loss_wm",
            "Avg_Val_CE", "Avg_Val_INV", "Avg_Val_Total",
        ]
        csv_out_path = log_dir / ("target_baseline_crafter.csv" if is_crafter else "target_baseline_minigrid.csv")

    file_exists = _ensure_baseline_csv(csv_out_path, csv_columns)

    # 3. Main Phase Loop (Iterating through tasks)
    n_phases = d_cfg.n_phases
    task_indices = range(d_cfg.start_idx, d_cfg.start_idx + n_phases)
    task_names = [f"{d_cfg.task_prefix}{i}" for i in task_indices]
    target_dataset_size = getattr(d_cfg, "target_dataset_size", 20000)
    force_recollect_per_task = True

    for i, current_task in enumerate(task_names):
        if is_bipedal:
            train_path = os.path.join(d_cfg.data_path, f"{current_task}{d_cfg.train_suffix}")
            should_recollect = True  # Bipedal Target Tasks are pre-collected uniformly
        else:
            train_path = os.path.join(d_cfg.data_path, f"{current_task}_seed{seed}{d_cfg.train_suffix}")
            should_recollect = force_recollect_per_task
        
        print(f"\n[PHASE {i+1}/{n_phases}] Training on: {current_task} (Seed: {seed})")
        if should_recollect or (not os.path.exists(train_path)):
            print(
                f"Collecting {target_dataset_size} interactions for {current_task} "
                f"(Seed: {seed})..."
            )
            from omegaconf import OmegaConf
            
            col_cfg = copy.deepcopy(cfg)
            if not hasattr(col_cfg, "env"):
                with open_dict(col_cfg):
                    col_cfg.env = OmegaConf.create({})
            base_collect_cfg = getattr(getattr(cfg, "env", None), "collect", None)
            save_coverage_visualize = bool(getattr(base_collect_cfg, "save_coverage_visualize", False))
            save_env_visualize = bool(getattr(base_collect_cfg, "save_env_visualize", False))
            with open_dict(col_cfg.env):
                col_cfg.env.env_type = domain_name  # Required by _finalize_and_save for visualization dispatch
                col_cfg.env.collect = OmegaConf.create({
                    "episodes": 1000000, 
                    "maximum_dataset_size": target_dataset_size, 
                    "data_type": getattr(cfg.env.collect, "data_type", "random") if hasattr(cfg, "env") and hasattr(cfg.env, "collect") else "random",
                    "data_folder": str(d_cfg.data_path),
                    "data_save_path": "",
                    "visualize_save_path": os.path.join(str(TRAINER_PATH), "logs", "dataset_visualization", "target", "bipedal" if is_bipedal else domain_name),
                    "env_visualize_save_path": os.path.join(str(TRAINER_PATH), "logs", "env_visualization", "target", "bipedal" if is_bipedal else domain_name),
                    "visualize_filename": f"{current_task}_random_coverage{'_seed'+str(seed) if not is_bipedal else ''}.png",
                    "env_visualize_filename": f"{current_task}_random_env{'_seed'+str(seed) if not is_bipedal else ''}.png",
                    "save_coverage_visualize": save_coverage_visualize,
                    "save_env_visualize": save_env_visualize
                })
            
            level_folder = "bipedal_walker" if is_bipedal else domain_name
            env_txt = _resolve_level_txt_path(level_folder, current_task)
            
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

            if domain_name == "minigrid":
                obs_arr = sub_data.get("obs", None)
                if obs_arr is None:
                    obs_arr = sub_data.get("a", None)
                current_shape = _extract_map_hw(obs_arr)
                if current_shape is not None:
                    if fisher_target_shape is None:
                        fisher_target_shape = current_shape
                    else:
                        grown_shape = (
                            max(fisher_target_shape[0], current_shape[0]),
                            max(fisher_target_shape[1], current_shape[1]),
                        )
                        if grown_shape != fisher_target_shape:
                            print(
                                f"  [Replay] Map shape grew from {fisher_target_shape} to {grown_shape}; "
                                "harmonizing existing Fisher buffer samples."
                            )
                            fisher_target_shape = grown_shape
                            changed = fisher_buffer.harmonize_buffer_map_shape(fisher_target_shape)
                            if changed > 0:
                                print(f"  [Replay] Harmonized {changed} replay sample fields to shape {fisher_target_shape}.")

            # --- A. Training Step ---
            replay_data = None
            if len(fisher_buffer) > 0:
                try:
                    replay_data = fisher_buffer.export_dict()
                except ValueError as e:
                    print(f"  [Warn] Replay export failed: {e}")
                    if domain_name == "minigrid" and fisher_target_shape is not None:
                        changed = fisher_buffer.harmonize_buffer_map_shape(fisher_target_shape)
                        if changed > 0:
                            print(f"  [Replay] Re-harmonized {changed} replay sample fields; retrying export.")
                        try:
                            replay_data = fisher_buffer.export_dict()
                        except ValueError as e2:
                            print(f"  [Warn] Replay export still failed after harmonization: {e2}")
                            replay_data = None
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
                
                val_summary = validate_on_all_targets(
                    cfg,
                    net,
                    d_cfg.data_path,
                    task_names,
                    d_cfg.val_suffix,
                    phase_name=f"target_iter_{i+1}",
                    VALID_TIMES=1,
                )

                if val_summary["valid_count"] > 0:
                    phase_metrics["target_val_avg_val_loss_wm"] = val_summary["avg_val_loss_wm"]
                    phase_metrics["Avg_Val_Total"] = val_summary["avg_val_loss_wm"]
                    
                    if is_bipedal:
                        phase_metrics["target_val_contact_acc"] = val_summary.get("contact_acc", 0.0)
                        phase_metrics["target_val_contact_bce"] = val_summary.get("contact_bce", 0.0)
                        print(f"    -> Results: Avg C_ACC = {phase_metrics['target_val_contact_acc']:.5f}, Avg C_BCE = {phase_metrics['target_val_contact_bce']:.5f}, Avg Total = {phase_metrics['Avg_Val_Total']:.5f}")
                    else:
                        phase_metrics["target_val_val_ce_loss"] = val_summary.get("terrain_loss", 0.0)
                        phase_metrics["target_val_val_inv_loss"] = val_summary.get("inventory_loss", 0.0)
                        phase_metrics["Avg_Val_CE"] = val_summary.get("terrain_loss", 0.0)
                        phase_metrics["Avg_Val_INV"] = val_summary.get("inventory_loss", 0.0)
                        print(f"    -> Results: Avg CE = {phase_metrics['Avg_Val_CE']:.5f}, Avg INV = {phase_metrics['Avg_Val_INV']:.5f}, Avg Total = {phase_metrics['Avg_Val_Total']:.5f}")
                
                # Append this sub-step's metrics to CSV
                pd.DataFrame([phase_metrics], columns=csv_columns).to_csv(
                    csv_out_path,
                    mode='a',
                    header=not file_exists,
                    index=False,
                )
                file_exists = True

        # 4. Finalize Phase: Archive to Replay Buffer
        print(f"  [Buffer] Archiving {current_task} data...")
        fisher_buffer.add_from_npz(
            train_path,
            current_sample_ratio=cfg.attention_model.current_sample_ratio,
            target_shape=fisher_target_shape if domain_name == "minigrid" else None,
        )
        torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Experiment Complete. CSV saved to: {csv_out_path}")

if __name__ == "__main__":
    run_target_baseline_experiment()
