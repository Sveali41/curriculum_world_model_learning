import os
import sys
import math
import torch
import numpy as np
import pandas as pd
import copy
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from hydra import initialize, compose

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WM_ROOT = os.path.join(PROJECT_ROOT, "wm")
WM_ROOT = os.environ.get("WM_ROOT", DEFAULT_WM_ROOT)
for import_root in (PROJECT_ROOT, WM_ROOT):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from modelBased.data.data_collect import data_collect
from modelBased.world_model import AttentionWM_training
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.common.artifacts import align_world_model_artifact_path
from trainer.common.paths import RESULTS_ROOT, VISUALIZATIONS_ROOT


def _normalize_train_api_result(train_api_output):
    if isinstance(train_api_output, tuple):
        return train_api_output[0]
    return train_api_output


def _append_result_row(csv_out_path, metrics):
    row_df = pd.DataFrame([metrics])
    write_header = not os.path.exists(csv_out_path)
    row_df.to_csv(csv_out_path, mode="a", header=write_header, index=False)


def _resolve_bipedal_dataset_path(cfg, domain_name):
    """Resolve dataset path using the unified naming rule, with legacy fallback."""
    task_group = str(cfg.domains[domain_name].task_group)
    task_name = str(cfg.domains[domain_name].task_name)
    data_type = str(getattr(cfg.env.collect, "data_type", "")).strip()
    if not data_type:
        data_type = "uniform" if task_group == "target" else "random"

    # If it is a target task, it should point to trainer/data/bipedalwalker/target_tasks/
    if task_group == "target":
        target_folder = str(getattr(cfg.domains[domain_name], "target_tasks_folder", 
            os.path.join(os.environ["TRAINER_PATH"], "data", domain_name, "target_tasks")))
        return os.path.join(target_folder, f"{task_name}_{data_type}.npz")

    primary_with_type = os.path.join(
        os.environ["TRAIN_DATASET_PATH"],
        f"bipedalwalker_{task_group}_{task_name}_{data_type}.npz",
    )
    if os.path.exists(primary_with_type):
        return primary_with_type

    primary = os.path.join(
        os.environ["TRAIN_DATASET_PATH"],
        f"bipedalwalker_{task_group}_{task_name}.npz",
    )
    if os.path.exists(primary):
        return primary

    legacy = os.path.join(os.environ["TRAIN_DATASET_PATH"], f"bipedalwalker_train_{task_name}.npz")
    if os.path.exists(legacy):
        print(f"[Path Fallback] Using legacy dataset path: {legacy}")
        return legacy

    return primary_with_type


def setup_env():
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    os.environ["TRAINER_PATH"] = os.path.join(PROJECT_ROOT, "trainer")
    wm_root = os.environ.get("WM_ROOT", DEFAULT_WM_ROOT)
    os.environ["WM_ROOT"] = wm_root
    os.environ["WORLD_MODEL_PATH"] = os.path.join(wm_root, "modelBased")
    os.environ["TRAIN_DATASET_PATH"] = os.path.join(wm_root, "modelBased/data/train_world_model")
    os.environ["MODEL_FPATH"] = os.path.join(wm_root, "modelBased/models")

def setup_config():
    # Use Hydra's compose API to properly load defaults (like config_ued)
    # config_path is relative to this .py file
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config_cl")
    
    # Use open_dict to allow setting new keys on a structured DictConfig
    with open_dict(cfg):
        cfg.PROJECT_ROOT = PROJECT_ROOT
        cfg.TRAIN_DATASET_PATH = os.environ["TRAIN_DATASET_PATH"]
    
    return cfg, None


def _collect_single_bipedal_dataset(cfg, domain_name, task_group, task_folder, task_name, data_type, dataset_size):
    cfg_collect = copy.deepcopy(cfg)
    target_path = os.path.join(
        os.environ["TRAIN_DATASET_PATH"],
        f"bipedalwalker_{task_group}_{task_name}_{data_type}.npz",
    )
    env_path = os.path.join(
        PROJECT_ROOT,
        "trainer",
        "level",
        "bipedal_walker",
        str(task_folder),
        f"{task_name}.txt",
    )
    with open_dict(cfg_collect):
        cfg_collect.domains[domain_name].task_group = str(task_group)
        cfg_collect.domains[domain_name].task_folder = str(task_folder)
        cfg_collect.domains[domain_name].task_name = str(task_name)
        cfg_collect.env.env_path = env_path
        cfg_collect.env.collect.data_type = str(data_type)
        cfg_collect.env.collect.mini_dataset_size = int(dataset_size)
        cfg_collect.env.collect.maximum_dataset_size = int(dataset_size)
        cfg_collect.env.collect.data_save_path = target_path
        # Keep episode budget high enough; collection stops at maximum_dataset_size.
        cfg_collect.env.collect.episodes = max(int(getattr(cfg_collect.env.collect, "episodes", 1000)), 1000)
        cfg_collect.env.collect.save_env_visualize = True
        cfg_collect.env.collect.save_coverage_visualize = True
        phase_kind = "target" if str(task_group) == "target" else "minitask"
        cfg_collect.env.collect.env_visualize_save_path = str(
            VISUALIZATIONS_ROOT / "environments" / phase_kind / "bipedal"
        )
        cfg_collect.env.collect.visualize_save_path = str(
            VISUALIZATIONS_ROOT / "datasets" / phase_kind / "bipedal"
        )
        cfg_collect.env.collect.env_visualize_filename = f"{task_name}_{data_type}_env.png"
        cfg_collect.env.collect.visualize_filename = f"{task_name}_{data_type}_coverage.png"

    print(
        f"[Collect] {task_group}/{task_name} ({data_type}) -> {dataset_size} samples\n"
        f"          env_path={env_path}\n"
        f"          save_path={target_path}"
    )
    data_collect(cfg_collect)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Bipedal collection finished but dataset was not saved: {target_path}")
    print(f"[Saved] {target_path}")
    return target_path


def _collect_bipedal_session_datasets(cfg, domain_name, domain_cfg, mode):
    mini_task_names = list(getattr(domain_cfg, "phase_task_names", []))
    per_task_size = 5000

    if mode == "random_baseline":
        target_size = per_task_size * max(len(mini_task_names), 1)
        return [
            _collect_single_bipedal_dataset(
                cfg=cfg,
                domain_name=domain_name,
                task_group=str(getattr(domain_cfg, "target_task_group", "target")),
                task_folder=str(getattr(domain_cfg, "target_task_folder", "target_tasks")),
                task_name=str(getattr(domain_cfg, "target_task_name")),
                data_type="random",
                dataset_size=target_size,
            )
        ]

    collected_paths = []
    for task_name in mini_task_names:
        collected_paths.append(
            _collect_single_bipedal_dataset(
                cfg=cfg,
                domain_name=domain_name,
                task_group=str(getattr(domain_cfg, "phase_task_group", "minitask")),
                task_folder=str(getattr(domain_cfg, "phase_task_folder", "minitasks")),
                task_name=str(task_name),
                data_type="random",
                dataset_size=per_task_size,
            )
        )
    return collected_paths

def collect_data():
    """
    Modified function to collect uniform data for 20 target tasks.
    """
    setup_env()
    cfg, _ = setup_config()
    align_world_model_artifact_path(cfg)
    
    domain_name = getattr(cfg, "domain", "crafter")
    domain_cfg = cfg.domains[domain_name]
    
    task_prefix = str(getattr(domain_cfg, "target_task_prefix", f"{domain_name}_target_task_"))
    configured_start = int(getattr(domain_cfg, "target_task_start_idx", 0))
    configured_count = int(getattr(domain_cfg, "target_task_count", 20))
    task_start = configured_start
    if domain_name == "crafter":
        # Since 1-5 already collected and 6-20 were newly generated
        task_start = max(task_start, 6)
    task_end = configured_start + configured_count

    tasks = [f"{task_prefix}{i}" for i in range(task_start, task_end)]
    target_tasks_folder = str(getattr(domain_cfg, "target_tasks_folder", f"{PROJECT_ROOT}/trainer/data/{domain_name}/target_tasks/"))
    os.makedirs(target_tasks_folder, exist_ok=True)

    level_domain_folder = "bipedal_walker" if domain_name == "bipedalwalker" else domain_name
    default_target_level_folder = "target_task" if domain_name == "minigrid" else "target_tasks"
    target_level_folder = str(
        getattr(domain_cfg, "target_level_folder", default_target_level_folder)
    )
    target_level_dir = os.path.join(
        PROJECT_ROOT, "trainer", "level", level_domain_folder, target_level_folder
    )
    missing_layouts = [
        os.path.join(target_level_dir, f"{task}.txt")
        for task in tasks
        if not os.path.isfile(os.path.join(target_level_dir, f"{task}.txt"))
    ]
    if missing_layouts:
        raise FileNotFoundError(
            "Target collection cannot start because layout files are missing: "
            + ", ".join(missing_layouts[:10])
        )

    print(
        f"[Collection Config] domain={domain_name} | tasks={tasks[0]}..{tasks[-1]} "
        f"({len(tasks)}) | layouts={target_level_dir}"
    )
    
    for task in tasks:
        print(f"\n[Collection] Processing Target Task: {task}")
        
        with open_dict(cfg):
            # Collect an explicit, fixed transition budget from many short
            # random-start rollouts. This makes the meaning of "30,000" exact
            # and avoids stopping after only a few long episodes.
            target_transitions = 30000
            cfg.env.collect.data_type = "uniform"
            cfg.env.collect.uniform_reset_steps = 50
            cfg.env.collect.episodes = math.ceil(
                target_transitions / int(cfg.env.collect.uniform_reset_steps)
            )
            cfg.env.collect.mini_dataset_size = target_transitions
            cfg.env.collect.maximum_dataset_size = target_transitions
            # Target validation should cover local transitions throughout the
            # large map instead of always starting from the layout's fixed S.
            # CustomMiniGridEnv treats S as an empty tile and samples uniformly
            # from all empty cells on every reset; direction remains random.
            cfg.env.collect.replace_start_with_empty = True
            
            # Resolve the checked domain-specific target layout.
            cfg.env.env_path = os.path.join(target_level_dir, f"{task}.txt")
            # Persist the exact task/layout identity in NPZ metadata instead of
            # falling back to an empty domain-level path.
            cfg.domains[domain_name].task_name = task
            cfg.domains[domain_name].layout_path = cfg.env.env_path
            
            # Use the configured suffix, defaulting to `_uniform.npz`.
            suffix = getattr(domain_cfg, "target_task_suffix", "_uniform.npz")
            cfg.env.collect.data_save_path = os.path.join(target_tasks_folder, f"{task}{suffix}")
            
            # Keep collection headless; the resulting NPZ is the required artifact.
            cfg.env.collect.save_env_visualize = False
            # Set visualization paths dynamically based on domain
            domain_suffix = "bipedal" if domain_name == "bipedalwalker" else domain_name
            cfg.env.collect.env_visualize_save_path = str(
                VISUALIZATIONS_ROOT / "environments" / "target" / domain_suffix
            )
            cfg.env.collect.env_visualize_filename = f"{task}_uniform_env.png"
            
            # Save only the collected-position coverage heatmap. Environment
            # screenshots and the other training visualizations stay disabled.
            cfg.env.collect.save_coverage_visualize = True
            cfg.env.collect.visualize_save_path = str(
                VISUALIZATIONS_ROOT / "datasets" / "target" / domain_suffix
            )
            cfg.env.collect.visualize_filename = f"{task}_uniform_coverage.png"
        
        # Call the actual data collection logic
        data_collect(cfg)
        saved = np.load(str(cfg.env.collect.data_save_path), allow_pickle=True)
        records = np.asarray(saved["f"], dtype=object).reshape(-1)
        missing_current = []
        missing_next = []
        inventory_changes = 0
        for index, value in enumerate(records):
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            record = value if isinstance(value, dict) else {}
            if "current_carrying_token" not in record:
                missing_current.append(index)
            if "next_carrying_token" not in record:
                missing_next.append(index)
            if (
                "current_carrying_token" in record
                and "next_carrying_token" in record
                and int(record["current_carrying_token"])
                != int(record["next_carrying_token"])
            ):
                inventory_changes += 1
        if missing_current or missing_next:
            raise ValueError(
                f"Collected dataset {cfg.env.collect.data_save_path} has invalid "
                "MiniGrid inventory metadata: "
                f"missing_current={missing_current[:10]}, "
                f"missing_next={missing_next[:10]}"
            )
        observations = np.asarray(saved["a"])
        positions = set()
        for observation in observations:
            hits = np.argwhere(observation[0] == 10)
            if len(hits):
                positions.add(tuple(map(int, hits[0])))
        with open(str(cfg.env.env_path), "r", encoding="utf-8") as layout_file:
            layout_rows = layout_file.read().split("\n\n", 1)[0].splitlines()
        empty_positions = {
            (row_index, column_index)
            for row_index, row in enumerate(layout_rows)
            for column_index, cell in enumerate(row)
            if cell in {"E", "S"}
        }
        empty_position_count = len(empty_positions)
        visited_empty_positions = positions.intersection(empty_positions)
        position_coverage = (
            len(visited_empty_positions) / empty_position_count
            if empty_position_count else 0.0
        )
        print(f"[Done] Saved UNIFORM dataset to {cfg.env.collect.data_save_path}")
        print(
            f"[Verified] {len(records)} transitions | "
            f"inventory_changes={inventory_changes} | "
            f"empty-position coverage={len(visited_empty_positions)}/{empty_position_count} "
            f"({position_coverage:.1%})"
        )

def run_experiment_session(mode="minitask", all_results=None):
    """
    Runs a 6-phase continual learning experiment.
    mode: "minitask" (01->06) or "random_baseline" (random sampling from target)
    """
    setup_env()
    cfg, _ = setup_config()
    
    if all_results is None:
        all_results = []

    domain_name = getattr(cfg, "domain", "crafter")
    domain_cfg = cfg.domains[domain_name]

    if domain_name == "bipedalwalker":
        _collect_bipedal_session_datasets(cfg, domain_name, domain_cfg, mode)

    # --- Phase Setup ---
    if domain_name == "bipedalwalker":
        if mode == "random_baseline":
            n_phases = 1
            task_names = [str(getattr(domain_cfg, "target_task_name"))]
            exp_label = "Bipedal_Target_Random_Baseline"
        else:
            task_names = list(getattr(domain_cfg, "phase_task_names", []))
            n_phases = len(task_names)
            exp_label = "Bipedal_Minitask_CL"
    else:
        n_phases = 6
        if mode == "minitask":
            task_files = [os.path.join(cfg.TRAIN_DATASET_PATH, f"crafter_minitask_0{i}.npz") for i in range(1, 7)]
            exp_label = "Minitask_CL"
        else:
            task_files = [os.path.join(cfg.TRAIN_DATASET_PATH, "crafter_target_task_diamond_random.npz")] * n_phases
            exp_label = "Random_Baseline"
        validation_dataset = os.path.join(cfg.TRAIN_DATASET_PATH, "crafter_target_task_diamond_uniform.npz")

    with open_dict(cfg):
        if domain_name == "bipedalwalker":
            cfg.domains[domain_name].task_group = str(getattr(domain_cfg, "target_task_group", "target"))
            cfg.domains[domain_name].task_folder = str(getattr(domain_cfg, "target_task_folder", "target_tasks"))
            cfg.domains[domain_name].task_name = str(getattr(domain_cfg, "target_task_name"))
            cfg.env.collect.data_type = "uniform"
            validation_dataset = _resolve_bipedal_dataset_path(cfg, domain_name)
        cfg.attention_model.validation_data_dir = validation_dataset
    
    # 3. Initialize model components for this session
    old_params = None
    fisher = None
    # Replay buffer shared across phases in this session
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)

    print(f"\n\n{'='*70}")
    print(f"### [START SESSION] Type: {exp_label}")
    print(f"{'='*70}\n")

    for i in range(n_phases):
        if domain_name == "bipedalwalker":
            if mode == "random_baseline":
                with open_dict(cfg):
                    cfg.domains[domain_name].task_group = str(getattr(domain_cfg, "target_task_group", "target"))
                    cfg.domains[domain_name].task_folder = str(getattr(domain_cfg, "target_task_folder", "target_tasks"))
                    cfg.domains[domain_name].task_name = str(getattr(domain_cfg, "target_task_name"))
                    cfg.env.collect.data_type = "random"
                task_file = _resolve_bipedal_dataset_path(cfg, domain_name)
                task_name = f"target_random_phase_{i+1}"
            else:
                phase_task_name = task_names[i]
                with open_dict(cfg):
                    cfg.domains[domain_name].task_group = str(getattr(domain_cfg, "phase_task_group", "minitask"))
                    cfg.domains[domain_name].task_folder = str(getattr(domain_cfg, "phase_task_folder", "minitasks"))
                    cfg.domains[domain_name].task_name = phase_task_name
                    cfg.env.collect.data_type = "random"
                task_file = _resolve_bipedal_dataset_path(cfg, domain_name)
                task_name = phase_task_name
        else:
            task_file = task_files[i]
            task_name = os.path.splitext(os.path.basename(task_file))[0]
        if mode == "random_baseline" and domain_name != "bipedalwalker":
            task_name = f"random_phase_{i+1}"

        print(f"\n[PHASE {i+1}/{n_phases}] Mode: {exp_label} | Training on: {task_file}")

        # --- Configure Training ---
        cfg_train = copy.deepcopy(cfg)
        with open_dict(cfg_train):
            cfg_train.attention_model.data_dir = task_file
            if domain_name == "bipedalwalker":
                cfg_train.attention_model.validation_data_dir = None
        
        # Export old samples for Replay only in continual-learning mode
        replay_data = None
        if mode != "random_baseline":
            replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None

        # --- Execute Training (EWC + Replay) ---
        train_res = _normalize_train_api_result(AttentionWM_training.train_api(
            cfg_train, 
            net=None, 
            old_params=old_params, 
            fisher=fisher, 
            replay_data=replay_data
        ))
        
        # Update weights and fisher info for next task
        old_params = train_res["old_params"]
        fisher = train_res["fisher"]

        # --- Post-Training Maintenance ---
        # 1. Add fresh samples into Fisher Replay Buffer only in continual-learning mode
        if mode != "random_baseline":
            print(f"[Buffer] Sampling from {task_file} for future replay...")
            fisher_buffer.add_from_npz(
                cfg_train.attention_model.data_dir,
                current_sample_ratio=cfg.attention_model.ewc_ratio,
            )

        # 2. CROSS-TASK VALIDATION (Zero-shot on the common target)
        print(f"--- Validating Phase {i+1} on Target Dataset ---")
        val_cfg = copy.deepcopy(cfg)
        with open_dict(val_cfg):
            val_cfg.attention_model.freeze_weight = True
            val_cfg.attention_model.data_dir = validation_dataset
            val_cfg.attention_model.validation_data_dir = None
        
        val_res = _normalize_train_api_result(AttentionWM_training.train_api(
            val_cfg, 
            net=None, 
            old_params=old_params, 
            fisher=fisher
        ))

        # --- Metrics Recording (Standardized Columns) ---
        target_metrics_dict = {}
        if isinstance(val_res.get("avg_val_loss"), list) and len(val_res["avg_val_loss"]) > 0:
            target_metrics_dict = val_res["avg_val_loss"][0]

        metrics = {
            "experiment_type": exp_label,
            "domain": domain_name,
            "phase": i + 1,
            "task": task_name,
            "train_best_val_loss": train_res.get("best_loss", 0.0),
            "target_val_avg_val_loss_wm": target_metrics_dict.get("avg_val_loss_wm", 0.0)
        }
        if domain_name == "bipedalwalker":
            for key in [
                "val/token_hull_pose_mse",
                "val/token_hull_vel_mse",
                "val/token_leg1_hip_mse",
                "val/token_leg1_knee_mse",
                "val/token_leg1_contact_bce",
                "val/token_leg1_contact_acc",
                "val/token_leg2_hip_mse",
                "val/token_leg2_knee_mse",
                "val/token_leg2_contact_bce",
                "val/token_leg2_contact_acc",
                "val/token_lidar_near_mse",
                "val/token_lidar_far_mse",
            ]:
                metrics[key.replace("val/", "target_val_")] = target_metrics_dict.get(key, 0.0)
        else:
            metrics["target_val_val_inv_loss"] = target_metrics_dict.get("val/inv_loss", 0.0)
            metrics["target_val_val_ce_loss"] = target_metrics_dict.get("val/ce_loss", 0.0)
        all_results.append(metrics)
        
        log_dir = Path(getattr(cfg, "cl_log_dir", RESULTS_ROOT / "continual_learning"))
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_out_path = str(log_dir / "cl_comparison_results.csv")
        _append_result_row(csv_out_path, metrics)
        
        if domain_name == "bipedalwalker":
            print(
                f"[Metrics] Phase {i+1} Target WM Loss: {metrics['target_val_avg_val_loss_wm']:.4f} | "
                f"leg1_contact_acc={metrics.get('target_val_token_leg1_contact_acc', 0.0):.4f} | "
                f"leg2_contact_acc={metrics.get('target_val_token_leg2_contact_acc', 0.0):.4f}"
            )
        else:
            print(f"[Metrics] Phase {i+1} Target CE Loss: {metrics['target_val_val_ce_loss']:.4f}")

    print(f"\n[SUCCESS] Finished {exp_label} session.\n")
    return all_results

if __name__ == "__main__":
    setup_env()
    # [SWITCH] collect_data() = batch collect uniform data for all target tasks
    # run_experiment_session() = run continual learning training session
    collect_data() # 1. this is for collecting data
    # run_experiment_session(mode="minitask", all_results=[]) # 2. this is for running continual learning training session
