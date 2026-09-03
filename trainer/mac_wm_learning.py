import sys
import os
ROOT_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WM_ROOT = os.path.join(ROOT_DIR, "wm")
if WM_ROOT not in sys.path:
    sys.path.insert(0, WM_ROOT)
if ROOT_DIR not in sys.path:
    sys.path.insert(1, ROOT_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"), override=False)
except ImportError:
    pass

os.environ.setdefault("PROJECT_ROOT", ROOT_DIR)
os.environ.setdefault("WM_ROOT", WM_ROOT)
os.environ.setdefault("ENV_PATH", os.path.join(ROOT_DIR, "level"))
os.environ.setdefault("WORLD_MODEL_PATH", os.path.join(WM_ROOT, "modelBased"))
os.environ.setdefault(
    "TRAIN_DATASET_PATH",
    os.path.join(WM_ROOT, "modelBased", "data", "train_world_model"),
)
os.environ.setdefault("MODEL_FPATH", os.path.join(WM_ROOT, "modelBased", "models"))
os.environ.setdefault("GENERATOR_PATH", os.path.join(ROOT_DIR, "generator"))
os.environ.setdefault("TRAINER_PATH", os.path.join(ROOT_DIR, "trainer"))

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
from modelBased.continue_learning.reservoir_buffer import ReservoirReplayBuffer
from modelBased.common.artifacts import align_world_model_artifact_path
from modelBased.exploration.minigrid_corpus import MiniGridCorpusWriter

from generator.generator_interface import GeneratorInterface
from trainer.common.paths import RESULTS_ROOT
from trainer.common.utils import (
    MINIGRID_VAL_LOSS_FIELDS,
    set_seed,
    validate_on_target_task,
    save_validation_csv,
    convert_trajectories_to_batch,
    minigrid_changed_fraction,
)


def _ensure_csv_header_compatible(csv_path: Path, expected_columns):
    """Start a fresh file when a previous experiment used another schema."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    expected_header = ",".join(expected_columns)
    try:
        with open(csv_path, "r", encoding="utf-8") as handle:
            current_header = handle.readline().strip()
    except OSError:
        current_header = ""
    if current_header == expected_header:
        return True

    backup_path = Path(f"{csv_path}.legacy_backup")
    suffix = 1
    while backup_path.exists():
        backup_path = Path(f"{csv_path}.legacy_backup{suffix}")
        suffix += 1
    os.replace(csv_path, backup_path)
    print(f"[Logger] CSV schema changed; old file backed up to {backup_path}")
    return False


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
    align_world_model_artifact_path(cfg)
    """
    UED Adversarial Training Loop.
    Integrates Generator (PPO), World Model (AttentionWM), and Continual Learning (Fisher Buffer).
    """

    # --------------------------------------
    # 1. Setup and initialization
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import csv

    # Logging and data paths
    log_dir = Path(getattr(cfg, "mac_results_dir", RESULTS_ROOT / "mac"))
    os.makedirs(log_dir, exist_ok=True)
    csv_path = log_dir / "ued_adversarial_log.csv"
    # Suffix used for ablation-specific CSV files
    ablation_suffix = ""
    if hasattr(cfg, "ablation") and cfg.ablation.type != "none":
        ablation_suffix = f"_{cfg.ablation.type}"
    
    # Suffix used for non-default validation metrics
    metric_suffix = ""
    if getattr(cfg.attention_model, "validation_metric", "mse") != "mse":
        metric_suffix = f"_{cfg.attention_model.validation_metric}"
        
    # Always construct the path so default and overridden cases share one code path.
    env_type = getattr(cfg.attention_model, "env_type", "minigrid")
    mask_suffix = f"_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"

    if env_type == "minigrid":
        summary_csv_path = log_dir / (
            f"minigrid_ued_results_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"
            f"_focal_reservoir_sa{ablation_suffix}.csv"
        )
    else:
        summary_csv_path = log_dir / f"{env_type}_ued_results{mask_suffix}{ablation_suffix}{metric_suffix}.csv"
    if ablation_suffix or metric_suffix or env_type != "crafter":
        print(f"[Log] CSV Path Adjusted: {summary_csv_path}")

    data_save_dir = Path(getattr(cfg.env.collect, "data_folder", str(TRAINER_PATH / "data")))
    
    # === A. Initialize the temporary data directory ===
    temp_data_dir = Path(
        getattr(
            cfg,
            "mac_temp_data_dir",
            str(TRAINER_PATH / "data" / env_type / "mini_tasks_temp" / "mac"),
        )
    )
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Store all UED-collected data in the dedicated temporary directory.
    cfg.env.collect.data_folder = str(temp_data_dir) + "/"

    
    # [TARGET DATA] Domain-specific fixed target datasets
    target_data_dir = data_save_dir
    domain_cfg = cfg.domains[env_type] if hasattr(cfg, "domains") and env_type in cfg.domains else None
    if domain_cfg is not None:
        target_data_dir = Path(
            getattr(
                domain_cfg,
                "val_data_path",
                getattr(domain_cfg, "target_tasks_folder", target_data_dir),
            )
        )
    # Remove stale temporary files from previous runs.
    for f in os.listdir(temp_data_dir):
        if f.endswith(".npz"): 
            try: os.remove(temp_data_dir / f)
            except: pass

    # === Initialize the summary CSV ===
    # MiniGrid's latent/transition diagnostics are part of its new schema.
    is_bipedal = (env_type == "bipedalwalker")
    is_minigrid = (env_type == "minigrid")
    if is_bipedal:
        csv_header = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_contact_acc", "gen_val_contact_bce", "gen_val_avg_val_loss_wm",
            "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
        ]
    elif is_minigrid:
        csv_header = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_avg_val_loss_wm", "target_val_avg_val_loss_wm",
            "target_val_valid_count",
            "target_val_focal_loss", "target_val_changed_focal_loss",
            "target_val_false_set_rate", "target_val_changed_count",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
            "Replay_Changed_Fraction", "Batch_Changed_Count",
            "Map_Novelty", "Combination_Novelty", "Random_Feature_Novelty",
            "Pre_Changed_Focal_Loss", "Post_Changed_Focal_Loss", "Learning_Progress",
            "Difficulty_Rank", "Learning_Progress_Rank", "Novelty_Rank", "Batch_Nearest_Hamming",
            "Archive_Nearest_Hamming", "Novelty_Distance_Std", "Latent_Batch_LogDet",
            "Mean_Object_Pair_Distance", "Mean_Nearest_Object_Distance",
            "Selected_Edit_Pair_Distance", "Mean_Edit_Rate", "Unique_Goal_Positions",
            "Reward_Learning_Progress", "Reward_Combination_Novelty",
            "Reward_Random_Feature_Novelty", "Final_Generator_Reward",
        ]
    else:
        csv_header = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_val_inv_loss", "gen_val_val_ce_loss", "gen_val_avg_val_loss_wm",
            "target_val_val_inv_loss", "target_val_val_ce_loss", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len", "Inv_Change_Ratio",
        ]

    file_exists = _ensure_csv_header_compatible(summary_csv_path, csv_header)
    if not file_exists:
        with open(summary_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            print(f"[Logger] Experiment summary initialized with {len(csv_header)} columns at {summary_csv_path}")
    else:
        print(f"[Logger] Reusing existing summary CSV: {summary_csv_path}")
    print(f"[Logger] Experiment summary will be saved to {summary_csv_path}")
    
    # === Domain-aware single-rollout transition cap for MAC/DR ===
    _apply_domain_collection_budget(cfg, env_type)

    # Optionally start from a clean checkpoint state.
    ckpt_path = cfg.attention_model.model_save_path
    if (
        getattr(cfg, "force_fresh_start", False)
        and is_minigrid
        and str(domain_cfg.exploration_policy).lower() == "rmax"
        and bool(getattr(domain_cfg.rmax_like, "resume", False))
    ):
        raise ValueError(
            "force_fresh_start=true is incompatible with "
            "domains.minigrid.rmax_like.resume=true"
        )
    if getattr(cfg, "force_fresh_start", False):
        checkpoint_paths = [ckpt_path]
        if is_minigrid and str(domain_cfg.exploration_policy).lower() == "rmax":
            checkpoint_paths.append(domain_cfg.rmax_like.checkpoint_path)
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"[Fresh Start] Deleted existing checkpoint: {checkpoint_path}")
            else:
                print(
                    "[Fresh Start] No checkpoint found to delete at: "
                    f"{checkpoint_path}"
                )

    # === A. Initialize the world model ===
    wm_instance = AttentionWorldModel(cfg.attention_model).to(device)

    # === B. Initialize the generator interface ===
    gen_interface = GeneratorInterface(
        world_model=wm_instance,
        device=device,
        cfg=cfg,
        agent_type=cfg.generator_agent.agent_type
    )

    # === C. Initialize the Fisher replay buffer ===
    if str(cfg.domain) == "minigrid":
        fisher_buffer = ReservoirReplayBuffer(
            max_size=cfg.attention_model.fisher_buffer_size,
            seed=int(getattr(cfg, "seed", 0)),
        )
    else:
        fisher_buffer = FisherReplayBuffer(
            max_size=cfg.attention_model.fisher_buffer_size,
            contact_positive_ratio=float(getattr(cfg.domains[cfg.domain], "contact_positive_ratio", 0.5)),
        )

    # === D. Training state variables ===
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
    corpus_writer = None
    if is_minigrid:
        explorer_ab_cfg = getattr(domain_cfg, "explorer_ab", None)
        corpus_path = getattr(explorer_ab_cfg, "corpus_export_path", None)
        if corpus_path:
            if str(domain_cfg.exploration_policy).lower() != "random":
                raise ValueError(
                    "MAC explorer corpus export requires "
                    "domains.minigrid.exploration_policy=random"
                )
            expected_size = int(explorer_ab_cfg.expected_corpus_size)
            actual_size = int(total_iterations) * int(cfg.generator_agent.batch_size)
            if actual_size != expected_size:
                raise ValueError(
                    f"corpus export expects {expected_size} maps, but "
                    f"iterations × batch size is {actual_size}"
                )
            corpus_writer = MiniGridCorpusWriter(
                corpus_path,
                expected_size,
                generation_seed=int(seed),
                generation_metadata={
                    "mac_iterations": int(total_iterations),
                    "generator_batch_size": int(cfg.generator_agent.batch_size),
                    "wm_epochs_per_iteration": int(cfg.attention_model.n_epochs),
                    "transitions_per_generated_map": int(
                        cfg.env.collect.maximum_dataset_size
                    ),
                },
            )
            print(f"[Explorer A/B] Will export frozen corpus to {corpus_writer.path}")
    warmup_iterations = _safe_int_cfg(
        getattr(cfg.generator_agent, "warmup_iterations", 0),
        default=0,
        name="generator_agent.warmup_iterations",
    )
    wm_train_frequency = cfg.generator_agent.wm_train_frequency  
    warmup_cleanup_done = False

    # === E. Validation set definition (fixed target tasks) ===
    if domain_cfg is not None and (
        hasattr(domain_cfg, "val_task_prefix")
        or hasattr(domain_cfg, "target_task_prefix")
    ):
        task_prefix = str(
            getattr(
                domain_cfg,
                "val_task_prefix",
                getattr(domain_cfg, "target_task_prefix", ""),
            )
        )
        task_suffix = str(
            getattr(
                domain_cfg,
                "val_suffix",
                getattr(domain_cfg, "target_task_suffix", "_uniform.npz"),
            )
        )
        task_start = int(
            getattr(
                domain_cfg,
                "val_start_idx",
                getattr(domain_cfg, "target_task_start_idx", 0),
            )
        )
        task_count = int(
            getattr(
                domain_cfg,
                "val_n_phases",
                getattr(domain_cfg, "target_task_count", 0),
            )
        )
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
    # 2. Main loop
    # --------------------------------------
    for iteration in range(total_iterations):
        print(
            f"\n=== Iteration {iteration + 1}/{total_iterations} ==="
        )
        inv_change_ratio = 0.0

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
        # Step 0.5: Periodically reset the diversity archive.
        # --------------------------------------------------------
        div_reset_interval = int(getattr(cfg.generator_agent, "div_reset_interval", 0))
        if (
            warmup_cleanup_done
            and env_type != "minigrid"
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
        # Step 1: Generator step (generate -> explore -> collect)
        # --------------------------------------------------------
        print(
            "[Generator] Generating environments and collecting trajectories..."
        )

        # Generator step returns the metric bundle used by training and logging.
        step_res = gen_interface.step(old_params=old_params, iteration=iteration)
        if corpus_writer is not None:
            generated_batch = gen_interface.last_generated_minigrid_batch
            if generated_batch is None:
                raise RuntimeError("generator did not expose its MiniGrid batch")
            corpus_writer.append_batch(**generated_batch)
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
            # Keep the update path active so the generator can learn from failures
            # signaled by negative reward.

        # --------------------------------------------------------
        # Step 2: Prepare buffer inputs
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
            # Include inventory tensors so replay-based world-model training
            # retains inventory supervision.
            if new_batch.get('rew') is not None:
                buffer_input["rew"] = new_batch["rew"]
            if new_batch.get('done') is not None:
                buffer_input["done"] = new_batch["done"]
            if new_batch.get('inv') is not None:
                buffer_input["inv"] = new_batch["inv"]
            if new_batch.get('inv_next') is not None:
                buffer_input["inv_next"] = new_batch["inv_next"]
                try:
                    inv_cur = new_batch.get("inv")
                    inv_nxt = new_batch.get("inv_next")
                    if inv_cur is not None and inv_nxt is not None:
                        if torch.is_tensor(inv_cur):
                            inv_cur = inv_cur.detach().cpu().numpy()
                        if torch.is_tensor(inv_nxt):
                            inv_nxt = inv_nxt.detach().cpu().numpy()
                        delta = np.abs(inv_nxt.astype(np.float32) - inv_cur.astype(np.float32))
                        inv_change_ratio = float((delta > 1e-6).mean())
                except Exception as e:
                    print(f"[Warning] Failed to compute Inv_Change_Ratio: {e}")
                    inv_change_ratio = 0.0
            # Buffer updates happen after training to avoid double counting.


        # PPO remains delayed until after WM training so MiniGrid can measure
        # held-out pre/post learning progress for this generated layout batch.
        gen_loss = 0.0
        gen_entropy = 0.0
        gen_mean_reward = 0.0

        # --------------------------------------------------------
        # Step 4: Update the world model
        # --------------------------------------------------------
        wm_final_loss = 0.0 # default if not trained
        
        # During warmup, skip world-model updates and data accumulation.
        is_warmup_for_wm = (iteration < warmup_iterations)
        
        if (not is_warmup_for_wm) and (iteration % wm_train_frequency == 0) and (new_batch is not None):
            print("[World Model] Retraining on current + replay data...")
            
            # Train on the full batch. Filtering is only applied during buffer updates.
            # if new_batch is not None:
            #     new_batch = filter_balanced_batch(
            #         new_batch, 
            #         fisher_buffer, 
            #         ratio=cfg.attention_model.current_sample_ratio, 
            #         elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio
            #     )
            
            # Save the full batch to a temporary dataset file.
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
            
            # 2. Get replay data from the buffer
            if len(fisher_buffer) > 0:
                replay_data = fisher_buffer.export_dict()
            else:
                replay_data = None
                print("[System] Fisher Buffer is empty. Training on current batch only.")

            # 3. Handle model freezing and reloading
            if old_params is None:
                # First training call or first call after warmup
                pass # Already handled by init
            
            # Unfreeze the model for training.
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = False 
            for param in wm_instance.parameters():
                param.requires_grad = True

            # 4. Train the world model.
            # Clear stale hooks to avoid dangling weak references.
            if hasattr(wm_instance, "_state_dict_hooks"):
                wm_instance._state_dict_hooks.clear()
            if hasattr(wm_instance, "_parameters"):
                for p_name, p in wm_instance._parameters.items():
                    if p is not None and hasattr(p, "_hooks"):
                        p._hooks.clear()

            # `train_api` returns `(result_dict, fisher, net)`.
            train_res, fisher, _ = AttentionWM_training.train_api(
                cfg,
                wm_instance, 
                old_params,
                fisher,
                replay_data=replay_data,
                # The canonical WM artifact is a full Lightning checkpoint,
                # so it can restore Adam, scheduler and global_step in the
                # next MAC iteration. The first update has no checkpoint yet.
                fit_ckpt_path=(
                    str(ckpt_path)
                    if bool(getattr(cfg.attention_model, "resume_optimizer", False))
                    and os.path.isfile(str(ckpt_path))
                    else None
                ),
            )
            # Update `old_params` for the next iteration.
            old_params = train_res.get("old_params")

            # Print explicit EWC-related metrics when W&B logging is disabled.
            ewc_term_val = train_res.get("ewc_term", train_res.get("train/ewc_term", None))
            loss_weighted_val = train_res.get("loss_weighted", train_res.get("train/loss_weighted", None))
            inv_loss_val = train_res.get("inv_loss", train_res.get("train/inv_loss", None))
            debug_mode = bool(getattr(cfg.attention_model, "debug_mode", False))
            if debug_mode and ((ewc_term_val is not None) or (loss_weighted_val is not None) or (inv_loss_val is not None)):
                print(
                    "[WM Metrics] "
                    f"ewc_term={float(ewc_term_val) if ewc_term_val is not None else float('nan'):.6f} | "
                    f"loss_weighted={float(loss_weighted_val) if loss_weighted_val is not None else float('nan'):.6f} | "
                    f"inv_loss={float(inv_loss_val) if inv_loss_val is not None else float('nan'):.6f}"
                )
            
            # Step 4.2: Delete temporary data after training
            if current_data_path and os.path.exists(current_data_path):
                try: os.remove(current_data_path)
                except: pass
            
            # Reuse validation loss later for logging.
            
            # Restore the freeze configuration.
            cfg.attention_model.freeze_weight = old_freeze

            # 5. Reload a clean model instance
            print("[System] Reloading model from checkpoint to clear hooks...")
            ckpt_path = cfg.attention_model.model_save_path
            wm_instance = AttentionWorldModel(cfg.attention_model).to(device)
            try:
                # Compatibility path for recent PyTorch checkpoint loading behavior.
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

            # Resynchronize the generator with the updated world model.
            gen_interface.sync_world_model(wm_instance.state_dict())

            print(
                "[System] World Model updated, reloaded, and synced to Generator."
            )

        # Evaluate held-out post-loss and apply LP + diversity before PPO.
        if is_minigrid:
            gen_interface.finalize_minigrid_rewards()
            gen_loss, gen_entropy, gen_mean_reward = gen_interface.update(iteration=iteration)
            print(
                f"[Generator] Policy Updated. Loss: {gen_loss:.4f} | "
                f"Entropy: {gen_entropy:.4f} | Mean reward: {gen_mean_reward:.4f}"
            )
        else:
            gen_loss, gen_entropy, gen_mean_reward = gen_interface.update(iteration=iteration)
            print(
                f"[Generator] Policy Updated. Loss: {gen_loss:.4f} | "
                f"Entropy: {gen_entropy:.4f} | Mean reward: {gen_mean_reward:.4f}"
            )

        # --------------------------------------------------------
        # Step 4.5: Update Fisher Buffer (Archive Current Data)
        # --------------------------------------------------------
        # We do this AFTER training so that 'replay_data' (used in training) 
        # strictly contains PAST data, while 'curr_data' contains CURRENT data.
        if buffer_input is not None and not is_warmup_for_wm:
             if str(cfg.domain) == "minigrid":
                 fisher_buffer.add_from_batch(buffer_input)
             else:
                 fisher_buffer.add_from_batch(
                    buffer_input,
                    current_sample_ratio=cfg.attention_model.current_sample_ratio,
                    fisher_buffer_elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio,
                )
             print(
                f"[Buffer] Archived {new_data_size} transitions. "
                f"Buffer Size: {len(fisher_buffer)}"
            )
        elif is_warmup_for_wm:
             print("[Buffer] Warmup phase: Skipping data accumulation to match budget.")
        
        # --------------------------------------------------------
        # Step 5: Validation and CSV logging
        # --------------------------------------------------------
        target_mean_loss = 0.0
        target_max_loss = 0.0
        target_std_loss = 0.0
        target_val_valid_count = 0
        target_val_field_losses = {
            name: float("nan") for name in MINIGRID_VAL_LOSS_FIELDS
        }
        target_val_focal_loss = 0.0
        target_val_changed_focal_loss = 0.0
        target_val_false_set_rate = 0.0
        target_val_changed_count = 0.0
        # Validation policy:
        # 1. Skip validation during early warmup to save time.
        # 2. Validate every step afterward to track progress.
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
            target_field_loss_values = {
                name: [] for name in MINIGRID_VAL_LOSS_FIELDS
            }
            target_focal_losses = []
            target_changed_focal_losses = []
            target_false_set_rates = []
            target_changed_counts = []
            
            # Temporarily switch to validation mode.
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = True

            # Disable W&B during validation to avoid hook errors and run spam.
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
                    if is_minigrid:
                        target_focal_losses.append(float(res_dict.get("focal_loss", 0.0)))
                        target_changed_focal_losses.append(float(res_dict.get("changed_focal_loss", 0.0)))
                        target_false_set_rates.append(float(res_dict.get("false_set_rate", 0.0)))
                        target_changed_counts.append(float(res_dict.get("changed_count", 0.0)))
                        for name in MINIGRID_VAL_LOSS_FIELDS:
                            value = float(res_dict.get(name, float("nan")))
                            if np.isfinite(value):
                                target_field_loss_values[name].append(value)
                    if is_bipedal:
                        target_contact_accs.append(res_dict.get('contact_acc', 0.0))
                        target_contact_bces.append(res_dict.get('contact_bce', 0.0))
                    elif not is_minigrid:
                        target_ce_losses.append(res_dict.get('terrain_loss', 0.0))
                        target_inv_losses.append(res_dict.get('inventory_loss', 0.0))
            
            # Restore configuration values.
            cfg.attention_model.freeze_weight = old_freeze
            cfg.attention_model.use_wandb = old_use_wandb

            # Aggregate target-task metrics.
            if target_avg_losses:
                target_val_valid_count = len(target_avg_losses)
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
                    target_val_focal_loss = float(np.mean(target_focal_losses)) if target_focal_losses else 0.0
                    target_val_changed_focal_loss = float(np.mean(target_changed_focal_losses)) if target_changed_focal_losses else 0.0
                    target_val_false_set_rate = float(np.mean(target_false_set_rates)) if target_false_set_rates else 0.0
                    target_val_changed_count = float(np.mean(target_changed_counts)) if target_changed_counts else 0.0
                    target_val_field_losses = {
                        name: (
                            float(np.mean(target_field_loss_values[name]))
                            if target_field_loss_values[name]
                            else float("nan")
                        )
                        for name in MINIGRID_VAL_LOSS_FIELDS
                    }
                    component_summary = " | ".join(
                        f"{name}: {value:.6f}"
                        for name, value in target_val_field_losses.items()
                    )
                    print(
                        f"[Metrics] Combined Target Loss -> Total: "
                        f"{target_val_avg_val_loss_wm:.6f} | {component_summary}"
                    )
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
        # Step 6: Write the experiment summary CSV
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
                            mg_metrics = getattr(gen_interface, "last_minigrid_metrics", {})
                            replay_metrics = fisher_buffer.export_dict() if len(fisher_buffer) else None
                            replay_changed_fraction, _ = minigrid_changed_fraction(replay_metrics)
                            _, batch_changed_count = minigrid_changed_fraction(buffer_input)
                            row_data = {
                                "Seed": seed,
                                "Iter": iteration + 1,
                                "Gen_Mean_Reward": f"{gen_mean_reward:.4f}",
                                "Gen_Loss": f"{gen_loss:.4f}",
                                "Gen_Entropy": f"{gen_entropy:.4f}",
                                "Gen_Div_Reward": f"{gen_div_reward_val:.4f}",
                                "gen_val_avg_val_loss_wm": f"{gen_val_avg_val_loss_wm:.6f}",
                                "target_val_avg_val_loss_wm": f"{target_val_avg_val_loss_wm:.6f}",
                                "target_val_valid_count": target_val_valid_count,
                                "target_val_focal_loss": f"{target_val_focal_loss:.6f}",
                                "target_val_changed_focal_loss": f"{target_val_changed_focal_loss:.6f}",
                                "target_val_false_set_rate": f"{target_val_false_set_rate:.6f}",
                                "target_val_changed_count": f"{target_val_changed_count:.2f}",
                                "New_Data_Size": new_data_size,
                                "Buffer_Size": len(fisher_buffer),
                                "Solvable_Count": f"{gen_solvable_count}",
                                "Avg_Path_Len": f"{gen_avg_bfs:.2f}",
                                "Replay_Changed_Fraction": f"{replay_changed_fraction:.6f}",
                                "Batch_Changed_Count": batch_changed_count,
                                "Map_Novelty": f"{mg_metrics.get('Map_Novelty', 0.0):.6f}",
                                "Combination_Novelty": f"{mg_metrics.get('Combination_Novelty', 0.0):.6f}",
                                "Random_Feature_Novelty": f"{mg_metrics.get('Random_Feature_Novelty', 0.0):.6f}",
                                "Pre_Changed_Focal_Loss": f"{mg_metrics.get('Pre_Changed_Focal_Loss', 0.0):.6f}",
                                "Post_Changed_Focal_Loss": f"{mg_metrics.get('Post_Changed_Focal_Loss', 0.0):.6f}",
                                "Learning_Progress": f"{mg_metrics.get('Learning_Progress', 0.0):.6f}",
                                "Difficulty_Rank": f"{mg_metrics.get('Difficulty_Rank', 0.0):.6f}",
                                "Learning_Progress_Rank": f"{mg_metrics.get('Learning_Progress_Rank', 0.0):.6f}",
                                "Novelty_Rank": f"{mg_metrics.get('Novelty_Rank', 0.0):.6f}",
                                "Batch_Nearest_Hamming": f"{mg_metrics.get('Batch_Nearest_Hamming', 0.0):.6f}",
                                "Archive_Nearest_Hamming": f"{mg_metrics.get('Archive_Nearest_Hamming', 0.0):.6f}",
                                "Novelty_Distance_Std": f"{mg_metrics.get('Novelty_Distance_Std', 0.0):.6f}",
                                "Latent_Batch_LogDet": f"{mg_metrics.get('Latent_Batch_LogDet', 0.0):.6f}",
                                "Mean_Object_Pair_Distance": f"{mg_metrics.get('Mean_Object_Pair_Distance', 0.0):.6f}",
                                "Mean_Nearest_Object_Distance": f"{mg_metrics.get('Mean_Nearest_Object_Distance', 0.0):.6f}",
                                "Selected_Edit_Pair_Distance": f"{mg_metrics.get('Selected_Edit_Pair_Distance', 0.0):.6f}",
                                "Mean_Edit_Rate": f"{mg_metrics.get('Mean_Edit_Rate', 0.0):.6f}",
                                "Unique_Goal_Positions": mg_metrics.get("Unique_Goal_Positions", 0),
                                "Reward_Learning_Progress": f"{mg_metrics.get('Reward_Learning_Progress', 0.0):.6f}",
                                "Reward_Combination_Novelty": f"{mg_metrics.get('Reward_Combination_Novelty', 0.0):.6f}",
                                "Reward_Random_Feature_Novelty": f"{mg_metrics.get('Reward_Random_Feature_Novelty', 0.0):.6f}",
                                "Final_Generator_Reward": f"{mg_metrics.get('Final_Generator_Reward', 0.0):.6f}",
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
                                "Avg_Path_Len": f"{gen_avg_ep_len:.2f}",
                                "Inv_Change_Ratio": f"{inv_change_ratio:.6f}"
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
        # Use the runtime collection folder (temp_data_dir) to avoid path drift.
        temp_files = glob.glob(str(temp_data_dir / f"UED_Dual_iter{iteration}_b*.npz"))
        if temp_files:
            print(f"[Cleanup] Deleting {len(temp_files)} temporary files for Iteration {iteration}...")
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"[Warning] Could not delete {f}: {e}")
            print(f"[Cleanup] Done.")

    if corpus_writer is not None:
        corpus_path = corpus_writer.finalize()
        print(f"[Explorer A/B] Frozen {corpus_writer.expected_size} maps at {corpus_path}")
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
    # Default entry point. Hydra reads ablation and seed settings from the config.
    # Use Hydra multirun to sweep across seeds or ablations.
    # python UED_wm_learning.py -m seed=0,1,2 ablation.type=none,no_diversity
    adversarial_ued_training_wrapper()
