import os
import sys
import torch
import numpy as np
import pandas as pd
import copy
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = "/home/siyao/phd_file/Research/rlPractice/Curriculum_world_model_learning"
sys.path.append(PROJECT_ROOT)

from modelBased.common.support import Support
from modelBased.data.data_collect import data_collect
from modelBased.world_model import AttentionWM_training
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer

def setup_env():
    os.environ["PROJECT_ROOT"] = PROJECT_ROOT
    os.environ["TRAINER_PATH"] = os.path.join(PROJECT_ROOT, "trainer")
    os.environ["WORLD_MODEL_PATH"] = os.path.join(PROJECT_ROOT, "modelBased")
    os.environ["TRAIN_DATASET_PATH"] = os.path.join(PROJECT_ROOT, "modelBased/data/train_world_model")
    os.environ["MODEL_FPATH"] = os.path.join(PROJECT_ROOT, "modelBased/models")

def setup_config():
    # Load the CL config
    config_path = os.path.join(PROJECT_ROOT, "trainer/conf/config_crafter_CL.yaml")
    cfg = OmegaConf.load(config_path)
    # Resolve interpolations
    cfg.PROJECT_ROOT = PROJECT_ROOT
    cfg.TRAIN_DATASET_PATH = os.environ["TRAIN_DATASET_PATH"]
    
    # Initialize support
    support = Support(cfg)
    return cfg, support

def collect_data():
    """
    Original function to collect data for each minitask.
    """
    setup_env()
    cfg, _ = setup_config()
    
    tasks = [f"crafter_minitask_0{i}" for i in range(1, 7)]
    # tasks.append("crafter_target_task_diamond") # optional
    
    for task in tasks:
        print(f"\n[Collection] Processing Task: {task}")
        
        # Set paths dynamically
        cfg.env.env_path = os.path.join(PROJECT_ROOT, f"trainer/level/crafter/{task}.txt")
        cfg.env.collect.data_save_path = os.path.join(os.environ["TRAIN_DATASET_PATH"], f"{task}.npz")
        cfg.env.collect.visualize_filename = f"{task}_random_coverage.png"
        
        # Call the actual data collection logic
        data_collect(cfg)
        print(f"[Done] Saved to {cfg.env.collect.data_save_path}")

def run_experiment_session(mode="minitask", all_results=None):
    """
    Runs a 6-phase continual learning experiment.
    mode: "minitask" (01->06) or "random_baseline" (random sampling from target)
    """
    setup_env()
    cfg, _ = setup_config()
    
    if all_results is None:
        all_results = []

    # --- Phase Setup ---
    n_phases = 6
    if mode == "minitask":
        task_files = [f"crafter_minitask_0{i}.npz" for i in range(1, 7)]
        exp_label = "Minitask_CL"
    else:
        # For random baseline, we always pull from the same random source file
        # The datamodule will trigger max_train_samples (20k) random sampling each time.
        task_files = ["crafter_target_task_diamond_random.npz"] * n_phases
        exp_label = "Random_Baseline"

    # 2. Preparation for Validation Dataset (The universal test target)
    validation_dataset = os.path.join(cfg.TRAIN_DATASET_PATH, "crafter_target_task_diamond_uniform.npz")
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
        task_filename = task_files[i]
        task_name = task_filename.split('.')[0]
        if mode == "random_baseline":
             task_name = f"random_phase_{i+1}"

        print(f"\n[PHASE {i+1}/n_phases] Mode: {exp_label} | Training on: {task_filename}")

        # --- Configure Training ---
        cfg_train = copy.deepcopy(cfg)
        cfg_train.attention_model.data_dir = os.path.join(cfg.TRAIN_DATASET_PATH, task_filename)
        
        # Export old samples for Replay
        replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None

        # --- Execute Training (EWC + Replay) ---
        train_res = AttentionWM_training.train_api(
            cfg_train, 
            net=None, 
            old_params=old_params, 
            fisher=fisher, 
            replay_data=replay_data
        )
        
        # Update weights and fisher info for next task
        old_params = train_res["old_params"]
        fisher = train_res["fisher"]

        # --- Post-Training Maintenance ---
        # 1. Add fresh samples into Fisher Replay Buffer
        print(f"[Buffer] Sampling from {task_filename} for future replay...")
        # target_shape should be large enough to handle ALL tasks (17x23 is the max)
        fisher_buffer.add_from_npz(
            cfg_train.attention_model.data_dir,
            current_sample_ratio=cfg.attention_model.ewc_ratio, # e.g., 0.05
            target_shape=(17, 23) 
        )

        # 2. CROSS-TASK VALIDATION (Zero-shot on the common uniform target)
        print(f"--- Validating Phase {i+1} on Target UNIFORM Dataset ---")
        val_cfg = copy.deepcopy(cfg)
        val_cfg.attention_model.freeze_weight = True
        val_cfg.attention_model.data_dir = validation_dataset
        
        val_res = AttentionWM_training.train_api(
            val_cfg, 
            net=None, 
            old_params=old_params, 
            fisher=fisher
        )

        # --- Metrics Recording (Standardized Columns) ---
        target_metrics_dict = {}
        if isinstance(val_res.get("avg_val_loss"), list) and len(val_res["avg_val_loss"]) > 0:
            target_metrics_dict = val_res["avg_val_loss"][0]

        metrics = {
            "experiment_type": exp_label,
            "phase": i + 1,
            "task": task_name,
            "train_best_val_loss": train_res.get("best_loss", 0.0),
            "target_val_val_inv_loss": target_metrics_dict.get("val/inv_loss", 0.0),
            "target_val_val_ce_loss": target_metrics_dict.get("val/ce_loss", 0.0),
            "target_val_avg_val_loss_wm": target_metrics_dict.get("avg_val_loss_wm", 0.0)
        }
        all_results.append(metrics)
        
        # Save to the specified combined CSV
        df = pd.DataFrame(all_results)
        csv_out_path = os.path.join(PROJECT_ROOT, "trainer/logs/cl_comparison_results.csv")
        df.to_csv(csv_out_path, index=False)
        
        print(f"[Metrics] Phase {i+1} Target CE Loss: {metrics['target_val_val_ce_loss']:.4f}")

    print(f"\n[SUCCESS] Finished {exp_label} session.\n")
    return all_results

if __name__ == "__main__":
    # # 1. Data Collection Mode
    # collect_data()
    
    # 2. Training Mode (A/B Comparison)
    setup_env()
    final_results = []
    # Start CL seq
    final_results = run_experiment_session(mode="minitask", all_results=final_results)
    # Start Random seq
    final_results = run_experiment_session(mode="random", all_results=final_results)