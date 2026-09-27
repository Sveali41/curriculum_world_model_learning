import csv
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
import json
import shutil
import hashlib
import time

# Add project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WM_ROOT = os.path.join(ROOT_DIR, "wm")
# Keep trainer results in the outer workspace even if a shell inherited a
# PROJECT_ROOT value pointing at the nested WM tree.
os.environ["TRAINER_ROOT"] = ROOT_DIR
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, WM_ROOT)

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

from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.world_model.crafter_event_audit import (
    inventory_event_episode_age_rows, inventory_event_rows,
)
from modelBased.world_model import AttentionWM_training
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.continue_learning.reservoir_buffer import ReservoirReplayBuffer
from modelBased.common.artifacts import align_world_model_artifact_path
from generator.generator_interface import GeneratorInterface
from trainer.common.run_resume import (
    config_digest, generator_state, guard_fresh_results, load_state, replay_state, restore_generator,
    restore_replay, restore_rng, rng_state, save_state, validate_sidecars,
)
from trainer.common.wm_snapshots import save_wm_update_snapshot
from trainer.common.utils import (
    CRAFTER_INVENTORY_VAL_METRICS,
    CRAFTER_FOCAL_VAL_METRICS,
    MINIGRID_VAL_LOSS_FIELDS,
    set_seed,
    validate_on_target_task,
    validate_on_all_targets,
    convert_trajectories_to_batch,
    minigrid_changed_fraction,
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


def _paired_batch_hash(batch):
    """Hash the transition fields consumed by WM training and replay."""
    digest = hashlib.sha256()
    for name in ("obs", "obs_next", "act", "rew", "done", "inv", "inv_next"):
        value = batch.get(name)
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _save_paired_batch(path: Path, batch):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: batch[name]
        for name in ("obs", "obs_next", "act", "rew", "done", "info", "inv", "inv_next")
        if batch.get(name) is not None
    }
    payload["batch_hash"] = np.asarray(_paired_batch_hash(batch))
    np.savez_compressed(path, **payload)
    return str(payload["batch_hash"].item())


def _load_paired_batch(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Paired DR batch is missing: {path}")
    with np.load(path, allow_pickle=True) as data:
        batch = {name: data[name] if name in data.files else None for name in ("obs", "obs_next", "act", "rew", "done", "info", "inv", "inv_next")}
        recorded_hash = str(data["batch_hash"].item()) if "batch_hash" in data.files else ""
    actual_hash = _paired_batch_hash(batch)
    if not recorded_hash or actual_hash != recorded_hash:
        raise ValueError(f"Paired DR batch hash mismatch for {path}: recorded={recorded_hash}, actual={actual_hash}")
    batch.update({"a": batch["obs"], "b": batch["obs_next"], "c": batch["act"], "d": batch["rew"], "e": batch["done"], "f": batch["info"], "g": batch["inv"], "h": batch["inv_next"]})
    return batch, actual_hash


def _append_crafter_event_rows(path: Path, seed: int, iteration: int, source: str, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(("Seed", "Iter", "Source", "Event_ID", "Delta_Vector",
                             "Count", "Changed_Slot_Count", "Changed_Slots"))
        writer.writerows((seed, iteration, source, row["event_id"], row["delta_vector"],
                          row["count"], row["changed_slot_count"], row["changed_slots"])
                         for row in rows)

def _append_crafter_episode_age_rows(path: Path, seed: int, iteration: int, rollouts):
    rows = []
    for rollout_index, rollout in enumerate(rollouts):
        if not all(rollout.get(key) is not None for key in ("inv", "inv_next", "done")):
            raise ValueError(f"Crafter rollout {rollout_index} lacks inventory/done fields for episode-age audit")
        rows.extend((rollout_index, row) for row in inventory_event_episode_age_rows(
            rollout["inv"], rollout["inv_next"], rollout["done"]
        ))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(("Seed", "Iter", "Rollout", "Episode_Age_Bin",
                             "Bin_Transition_Count", "Episodes_With_Transitions_In_Bin",
                             "Event_ID", "Delta_Vector", "Event_Count", "Rate_Per_1000",
                             "Changed_Slot_Count", "Changed_Slots"))
        for rollout_index, row in rows:
            writer.writerow((seed, iteration, rollout_index, row["episode_age_bin"],
                             row["bin_transition_count"], row["episodes_with_transitions_in_bin"],
                             row["event_id"], row["delta_vector"], row["event_count"],
                             f"{row['rate_per_1000']:.8f}", row["changed_slot_count"], row["changed_slots"]))

def _append_crafter_target_confusion(path: Path, seed: int, iteration: int, payload):
    if not payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(("Seed", "Iter", "True_Event_ID", "True_Delta_Vector",
                             "True_Count", "Predicted_Exact_Count", "Predicted_KEEP_Count",
                             "Predicted_Other_Count", "True_Change_Margin_Mean"))
        keep_count = int(payload.get("true_keep_count", 0))
        keep_false_positive = int(payload.get("true_keep_false_positive", 0))
        writer.writerow((seed, iteration, 0, json.dumps([0] * 12, separators=(",", ":")),
                         keep_count, keep_count - keep_false_positive,
                         keep_count - keep_false_positive, keep_false_positive, float("nan")))
        for event in payload.get("events", []):
            event_id = event.get("event_id")
            if event_id is None:
                event_id = -1
            writer.writerow((seed, iteration, event_id,
                             json.dumps(event.get("delta", ()), separators=(",", ":")),
                             event["count"], event["exact"], event["predicted_keep"],
                             event["predicted_other"],
                             event["margin_sum"] / event["margin_count"]
                             if event["margin_count"] else float("nan")))


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


def _crafter_gate_artifact_suffix(cfg: DictConfig, is_crafter: bool) -> str:
    """Return a stable artifact namespace for Crafter gate-loss variants."""
    if not is_crafter:
        return ""
    output_mode = str(
        getattr(cfg.attention_model, "crafter_inventory_output_mode", "categorical_gate")
    ).strip().lower()
    if output_mode == "categorical_effect":
        reduction = str(
            getattr(
                cfg.attention_model,
                "crafter_inventory_effect_reduction",
                "balanced_mean",
            )
        ).strip().lower()
        labels = {
            "balanced_mean": "effect5balanced",
            "sqrt_balanced": "effect5sqrt",
            "mean": "effect5natural",
        }
        label = labels.get(reduction)
        if label is None:
            label = "effect5_" + "".join(
                ch if ch.isalnum() else "_" for ch in reduction
            ).strip("_")
        return f"_{label}"
    gate_mode = str(
        getattr(cfg.attention_model, "crafter_inventory_gate_reduction", "global_balanced")
    ).strip().lower()
    labels = {
        "global_balanced": "globalgate",
        "slot_macro_changed": "slotmacro",
    }
    label = labels.get(gate_mode)
    if label is None:
        label = "".join(ch if ch.isalnum() else "_" for ch in gate_mode).strip("_")
    return f"_{label or 'unspecifiedgate'}"


def _crafter_pose_artifact_suffix(cfg: DictConfig, is_crafter: bool) -> str:
    """Keep opt-in learned-pose experiments out of legacy artifact names."""
    if not is_crafter:
        return ""
    pose_cfg = getattr(cfg.attention_model, "crafter_pose", None)
    enabled = (
        bool(pose_cfg.get("enabled", False)) if isinstance(pose_cfg, dict)
        else bool(getattr(pose_cfg, "enabled", False)) if pose_cfg is not None else False
    )
    return "_pose" if enabled else ""


def _crafter_event_residual_artifact_suffix(cfg: DictConfig, is_crafter: bool) -> str:
    """Keep opt-in legal-event residual runs separate from old DR artifacts."""
    if not is_crafter:
        return ""
    residual = getattr(cfg.attention_model, "crafter_inventory_event_residual", None)
    enabled = bool(residual.get("enabled", False)) if isinstance(residual, dict) else bool(
        getattr(residual, "enabled", False)
    ) if residual is not None else False
    if not enabled:
        return ""
    dims = residual.get("hidden_dims", (64,)) if isinstance(residual, dict) else getattr(residual, "hidden_dims", (64,))
    label = "x".join(str(int(width)) for width in dims)
    return f"_eventresidual{label}"


def _crafter_validation_artifact_suffix(val_suffix: str, is_crafter: bool) -> str:
    """Isolate the new Crafter coverage-v3 experiment from legacy artifacts."""
    if not is_crafter:
        return ""
    return "_coverage_v3" if Path(str(val_suffix)).name == "_coverage_v3.npz" else ""


@hydra.main(version_base=None, config_path="conf", config_name="config_dr")
def run_dr_baseline_experiment(cfg: DictConfig):
    align_world_model_artifact_path(cfg)
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
    is_crafter = (domain_name == "crafter")
    val_n_phases = int(getattr(d_cfg, "val_n_phases", getattr(d_cfg, "target_task_count", 20)))
    val_task_prefix = str(getattr(d_cfg, "val_task_prefix", getattr(d_cfg, "target_task_prefix", "target_task")))
    val_data_path = str(getattr(d_cfg, "val_data_path", getattr(d_cfg, "target_tasks_folder", "")))
    val_suffix = str(getattr(d_cfg, "val_suffix", getattr(d_cfg, "target_task_suffix", "_uniform.npz")))
    crafter_validation_suffix = _crafter_validation_artifact_suffix(val_suffix, is_crafter)
    val_start_idx = int(
        getattr(
            d_cfg,
            "val_start_idx",
            getattr(d_cfg, "start_idx", getattr(d_cfg, "target_task_start_idx", 0)),
        )
    )
    
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

    transition_replay_cfg = getattr(cfg.attention_model, "crafter_transition_replay", None)
    if isinstance(transition_replay_cfg, dict):
        transition_replay_enabled = is_crafter and bool(transition_replay_cfg.get("enabled", False))
        configured_include_changed_slot = bool(
            transition_replay_cfg.get("include_changed_slot", False)
        )
    else:
        transition_replay_enabled = is_crafter and bool(getattr(transition_replay_cfg, "enabled", False))
        configured_include_changed_slot = bool(
            getattr(transition_replay_cfg, "include_changed_slot", False)
        )
    transition_replay_include_changed_slot = (
        transition_replay_enabled and configured_include_changed_slot
    )
    crafter_value_mode = str(
        getattr(cfg.attention_model, "crafter_inventory_value_mode", "categorical_absolute")
    ).lower()
    crafter_inventory_output_mode = str(
        getattr(cfg.attention_model, "crafter_inventory_output_mode", "categorical_gate")
    ).lower()
    crafter_value_mode_suffix = (
        "_delta"
        if is_crafter
        and crafter_inventory_output_mode == "categorical_gate"
        and crafter_value_mode == "categorical_delta"
        else ""
    )
    protected_cfg = getattr(transition_replay_cfg, "protected_slot_replay", {})
    protected_enabled = bool(
        protected_cfg.get("enabled", False) if isinstance(protected_cfg, dict)
        else getattr(protected_cfg, "enabled", False)
    )
    crafter_gate_suffix = _crafter_gate_artifact_suffix(cfg, is_crafter)
    crafter_pose_suffix = _crafter_pose_artifact_suffix(cfg, is_crafter)
    crafter_event_residual_suffix = _crafter_event_residual_artifact_suffix(cfg, is_crafter)

    # Optionally start from a clean checkpoint state.
    ckpt_path = cfg.attention_model.model_save_path
    force_fresh_start = bool(getattr(cfg, "force_fresh_start", False))
    resume_training = is_crafter and bool(getattr(cfg, "resume_training", False))
    if resume_training and force_fresh_start:
        raise ValueError("resume_training=true requires force_fresh_start=false")
    uses_minigrid_rmax = (
        domain_name == "minigrid"
        and str(d_cfg.exploration_policy).lower() == "rmax"
    )
    if (
        force_fresh_start
        and uses_minigrid_rmax
        and bool(getattr(d_cfg.rmax_like, "resume", False))
    ):
        raise ValueError(
            "force_fresh_start=true is incompatible with "
            "domains.minigrid.rmax_like.resume=true"
        )
    if is_crafter and not resume_training and bool(getattr(cfg, "save_iteration_state", True)):
        ablation_type_for_guard = str(getattr(getattr(cfg, "ablation", None), "type", "none"))
        ablation_suffix_for_guard = "" if ablation_type_for_guard == "none" else f"_{ablation_type_for_guard}"
        guard_fresh_results(
            Path(cfg.dr_log_dir) / f"dr_crafter_results{ablation_suffix_for_guard}.csv",
            Path(cfg.dr_log_dir) / f"crafter_dr_seed{seed}.resume.pt", seed,
        )
    if force_fresh_start:
        checkpoint_paths = [ckpt_path]
        if uses_minigrid_rmax:
            checkpoint_paths.append(d_cfg.rmax_like.checkpoint_path)
        for checkpoint_path in checkpoint_paths:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"[Fresh Start] Deleted existing checkpoint: {checkpoint_path}")
            else:
                print(
                    "[Fresh Start] No checkpoint found to delete at: "
                    f"{checkpoint_path}"
                )

    # 1. Initialization
    wm = AttentionWorldModel(cfg.attention_model).to(device)

    # DR usually generates random maps via GeneratorInterface (agent_type='random')
    generator = GeneratorInterface(wm, device, cfg, agent_type='random')
    if str(cfg.domain) == "minigrid":
        fisher_buffer = ReservoirReplayBuffer(
            max_size=cfg.attention_model.fisher_buffer_size,
            seed=int(getattr(cfg, "seed", 0)),
        )
    else:
        fisher_buffer = FisherReplayBuffer(
            max_size=cfg.attention_model.fisher_buffer_size,
            contact_positive_ratio=float(getattr(cfg.domains[cfg.domain], "contact_positive_ratio", 0.5)),
            crafter_transition_replay=transition_replay_cfg if transition_replay_enabled else None,
            seed=int(getattr(cfg, "seed", 0)),
        )
    
    log_dir = Path(cfg.dr_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    temp_data_dir = Path(cfg.dr_temp_data_dir)
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # Store all DR-collected data in the dedicated temporary directory.
    cfg.env.collect.data_folder = str(temp_data_dir) + "/"

    mask_suffix = f"_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"
    if is_minigrid:
        summary_csv_path = log_dir / (
            f"dr_summary_minigrid_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"
            "_focal_reservoir.csv"
        )
    elif is_crafter:
        ablation_type = str(getattr(getattr(cfg, "ablation", None), "type", "none"))
        ablation_suffix = "" if ablation_type == "none" else f"_{ablation_type}"
        summary_csv_path = log_dir / f"dr_crafter_results{ablation_suffix}.csv"
    else:
        ewc_suffix = "_ewc_balanced_inventory" if bool(getattr(cfg.attention_model, "ewc_enabled", False)) else ""
        ewc_suffix += crafter_value_mode_suffix
        if transition_replay_enabled:
            ewc_suffix += "_transition_replay"
        if transition_replay_include_changed_slot:
            ewc_suffix += "_slot"
        if protected_enabled:
            ewc_suffix += "protect"
        ewc_suffix += crafter_gate_suffix
        ewc_suffix += crafter_pose_suffix
        ewc_suffix += crafter_event_residual_suffix
        ewc_suffix += crafter_validation_suffix
        summary_csv_path = log_dir / f"dr_summary_{domain_name}{mask_suffix}{ewc_suffix}.csv"
    transition_stats_csv_path = (
        log_dir / f"dr_crafter_replay{ablation_suffix}_seed{seed}.csv"
        if transition_replay_enabled else None
    )
    resume_state_path = log_dir / f"crafter_dr_seed{seed}.resume.pt"
    file_exists = False

    old_params, fisher = None, None
    global_best_selection = float("inf")
    global_best_iteration = 0
    def _transition_stat_columns(source, stats):
        """Flatten four transition classes for the compact DR summary."""
        result = {}
        total = sum(value["count"] for value in stats.values()) if stats else 0
        for transition_type in FisherReplayBuffer.CRAFTER_TRANSITION_TYPES:
            value = (stats or {}).get(transition_type, {"count": 0, "actions": {}})
            count = int(value["count"])
            result[f"{source}_{transition_type}_count"] = count
            result[f"{source}_{transition_type}_fraction"] = count / total if total else 0.0
            result[f"{source}_{transition_type}_action_coverage"] = len(value["actions"])
        return result
    def _slot_counts(stats):
        result = {}
        for values in (stats or {}).values():
            for slot, count in values.get("changed_slots", {}).items():
                result[int(slot)] = result.get(int(slot), 0) + int(count)
        return result
    def _sum_slot_counts(*rows):
        result = {}
        for row in rows:
            for slot, count in (row or {}).items():
                result[int(slot)] = result.get(int(slot), 0) + int(count)
        return result
    quick_target_loss_csv = is_crafter and bool(
        getattr(getattr(cfg, "dr_fast_ablation", None), "target_loss_csv", False)
    )
    quick_ewc_diagnostics_csv = is_crafter and bool(
        getattr(getattr(cfg, "dr_fast_ablation", None), "ewc_diagnostics_csv", False)
    )
    validation_slot_counts = {}
    if is_crafter and getattr(cfg.attention_model, "dr_validation_archive", None) and os.path.isfile(str(cfg.attention_model.dr_validation_archive)):
        validation_archive = np.load(str(cfg.attention_model.dr_validation_archive), allow_pickle=True)
        val_inv = validation_archive["g"] if "g" in validation_archive.files else validation_archive["inv"]
        val_next = validation_archive["h"] if "h" in validation_archive.files else validation_archive["inv_next"]
        changed = np.asarray(val_inv)[:, 4:16] != np.asarray(val_next)[:, 4:16]
        validation_slot_counts = {int(slot + 4): int(changed[:, slot].sum()) for slot in range(12) if changed[:, slot].any()}
    if is_minigrid:
        csv_columns = [
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
    elif not is_bipedal: # Crafter or others
        csv_columns = [
            "Seed", "Iter", "New_Data_Size", "Cumulative_Transitions", "Buffer_Size",
            "target_val_valid_count", "target_val_sample_count", "target_val_avg_val_loss_wm",
            "target_val_changed_focal_loss",
            "target_val_inventory_effect_false_positive_rate",
            "target_val_inventory_effect_change_recall",
            "target_val_inventory_effect_change_precision",
            "target_val_inventory_effect_row_exact",
            "paired_batch_hash",
            "target_val_layout_changed_focal_loss", "target_val_layout_false_set_rate", "target_val_layout_changed_count",
            "target_val_inventory_changed_focal_loss", "target_val_inventory_false_set_rate", "target_val_inventory_changed_count",
            "target_val_joint_accuracy", "target_val_position_accuracy", "target_val_direction_accuracy",
            "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward", "Inv_Change_Ratio",
            "Pre_Changed_Focal_Loss", "Post_Changed_Focal_Loss", "Learning_Progress",
            "Solvable_Count", "Avg_Path_Len",
            "Inventory_KEEP_Ratio", "Inventory_Stage_0_Count", "Inventory_Stage_0_Mean_LP", "Inventory_Stage_0_Reward_Std", "Inventory_Stage_1_Count", "Inventory_Stage_1_Mean_LP", "Inventory_Stage_1_Reward_Std", "Inventory_Stage_2_Count", "Inventory_Stage_2_Mean_LP", "Inventory_Stage_2_Reward_Std", "Inventory_Stage_3_Count", "Inventory_Stage_3_Mean_LP", "Inventory_Stage_3_Reward_Std", "Inventory_Stage_4_Count", "Inventory_Stage_4_Mean_LP", "Inventory_Stage_4_Reward_Std",
        ]
    elif is_bipedal:
        csv_columns = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_contact_acc", "gen_val_contact_bce", "gen_val_avg_val_loss_wm",
            "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
        ]
    if quick_target_loss_csv:
        insert_at = csv_columns.index("target_val_changed_focal_loss")
        csv_columns[insert_at:insert_at] = [
            "target_val_terrain_loss", "target_val_inventory_loss",
            "target_val_changed_nll",
        ]
    if quick_ewc_diagnostics_csv:
        csv_columns.extend((
            "train_ewc_raw_epoch", "train_ewc_weighted_epoch",
            "train_ewc_to_wm_ratio_epoch",
        ))
    file_exists = (
        summary_csv_path.is_file()
        if resume_training else _ensure_csv_header_compatible(summary_csv_path, csv_columns)
    )
    event_audit_cfg = getattr(cfg, "crafter_inventory_event_audit", None)
    event_audit_enabled = bool(
        event_audit_cfg.get("enabled", False) if isinstance(event_audit_cfg, dict)
        else getattr(event_audit_cfg, "enabled", False)
    ) and is_crafter
    event_audit_csv_path = summary_csv_path.with_name(f"{summary_csv_path.stem}.inventory_events.csv")
    episode_age_audit_csv_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.inventory_events_by_episode_age.csv"
    )
    target_confusion_csv_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.target_event_confusion.csv"
    )
    if event_audit_enabled and not bool(getattr(cfg.attention_model, "crafter_event_confusion_enabled", False)):
        raise ValueError("Crafter inventory event audit requires attention_model.crafter_event_confusion_enabled=true")

    # 1.5 Load existing model if available (Resume logic)
    ckpt_path = cfg.attention_model.model_save_path
    require_resume_checkpoint = bool(getattr(getattr(cfg, "dr_fast_ablation", None), "require_resume_checkpoint", False))
    if resume_training:
        pass  # Load the full iteration state after all state holders exist.
    elif force_fresh_start:
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
                if require_resume_checkpoint:
                    raise RuntimeError(f"Quick DR initial checkpoint could not be loaded: {ckpt_path}") from e
                print(f"[Warning] Failed to load existing model: {e}. Starting from scratch.")
        else:
            if require_resume_checkpoint:
                raise FileNotFoundError(f"Quick DR initial checkpoint is missing: {ckpt_path}")
            print(f"[System] No existing checkpoint found at {ckpt_path}. Starting from scratch.")

    fast_cfg = getattr(cfg, "dr_fast_ablation", None)
    paired_data_mode = str(getattr(fast_cfg, "paired_data_mode", "off")).lower()
    if paired_data_mode not in {"off", "write", "read"}:
        raise ValueError("dr_fast_ablation.paired_data_mode must be off, write, or read")
    if paired_data_mode != "off" and not is_crafter:
        raise ValueError("Paired batch mode is only supported for Crafter DR")
    paired_data_dir = Path(str(getattr(fast_cfg, "paired_data_dir", ""))) if paired_data_mode != "off" else None
    paired_train_seed = bool(getattr(fast_cfg, "paired_train_seed", False))
    if paired_train_seed and paired_data_mode == "off":
        raise ValueError("dr_fast_ablation.paired_train_seed requires paired_data_mode=write or read")

    # 2. Main Loop
    cumulative_transitions = 0
    start_iteration = 0
    if resume_training:
        state = load_state(resume_state_path, cfg, "crafter_dr", summary_csv_path, csv_columns)
        validate_sidecars(
            (event_audit_csv_path, episode_age_audit_csv_path, target_confusion_csv_path,
             *([transition_stats_csv_path] if transition_stats_csv_path is not None else [])),
            (), seed, state["completed_iteration"],
        )
        wm.load_state_dict(state["wm"])
        old_params, fisher = state["old_params"], state["fisher"]
        restore_replay(fisher_buffer, state["replay"])
        restore_generator(generator, state["generator"], include_policy=False)
        global_best_selection = state["global_best_selection"]
        global_best_iteration = state["global_best_iteration"]
        cumulative_transitions = state["cumulative_transitions"]
        start_iteration = state["completed_iteration"]
        restore_rng(state["rng"])
        print(f"[Resume] Crafter DR completed iteration {start_iteration}; continuing at {start_iteration + 1}")
    for iteration in range(start_iteration, cfg.generator_agent.total_iterations):
        print(f"\n>>> DR Iteration {iteration + 1}/{cfg.generator_agent.total_iterations}")
        wm_train_seconds = 0.0
        target_val_seconds = 0.0

        # A. Collect or reuse the exact same transition batch for both bias arms.
        paired_batch_hash = ""
        new_batch = None
        if paired_data_mode == "read":
            batch_path = paired_data_dir / f"seed{seed}_iter{iteration + 1:02d}.npz"
            new_batch, paired_batch_hash = _load_paired_batch(batch_path)
            print(f"  [Paired Data] Reused {len(new_batch['obs'])} rows from {batch_path}; hash={paired_batch_hash[:12]}")
            gen_val_avg_val_loss_wm = float("nan")
            gen_val_val_ce_loss = float("nan")
            gen_val_val_inv_loss = float("nan")
            gen_val_contact_acc = 0.0
            gen_val_contact_bce = 0.0
            gen_div_reward = 0.0
            valid_trajs = []
            solvable_count = 0
            avg_path_len = 0.0
        else:
            trajs = generator.step(old_params=old_params, iteration=iteration)
            # GeneratorInterface.step() returns a 9-tuple.
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
            if event_audit_enabled:
                _append_crafter_event_rows(
                    event_audit_csv_path, seed, iteration + 1, "new_batch",
                    inventory_event_rows(new_batch["inv"], new_batch["inv_next"]),
                )
                _append_crafter_episode_age_rows(
                    episode_age_audit_csv_path, seed, iteration + 1, valid_trajs,
                )
            if paired_data_mode == "write":
                batch_path = paired_data_dir / f"seed{seed}_iter{iteration + 1:02d}.npz"
                paired_batch_hash = _save_paired_batch(batch_path, new_batch)
                print(f"  [Paired Data] Saved batch hash={paired_batch_hash[:12]} to {batch_path}")

        if new_batch is None or new_batch.get("obs") is None:
            raise RuntimeError(f"No transition batch available for DR iteration {iteration + 1}")
        current_transitions = len(new_batch["obs"]) if new_batch is not None and new_batch.get("obs") is not None else 0
        inv_change_ratio = 0.0
        if is_crafter:
            inv_cur, inv_next = new_batch.get("inv"), new_batch.get("inv_next")
            if inv_cur is not None and inv_next is not None:
                inv_cur = inv_cur.detach().cpu().numpy() if torch.is_tensor(inv_cur) else np.asarray(inv_cur)
                inv_next = inv_next.detach().cpu().numpy() if torch.is_tensor(inv_next) else np.asarray(inv_next)
                inv_change_ratio = float((np.abs(inv_next.astype(np.float32) - inv_cur.astype(np.float32))[:, 4:16] > 1e-6).mean())
        cumulative_transitions += int(current_transitions)
        print(f"  [Data] Current batch transitions: {current_transitions}")
        current_transition_stats = (
            fisher_buffer.transition_replay_stats(new_batch)
            if transition_replay_enabled else None
        )
        sampled_replay_stats = None
        protected_replay_slot_counts = {}
        fisher_slot_counts = {}
        train_ewc_raw = float("nan")
        train_ewc_weighted = float("nan")
        train_ewc_to_wm_ratio = float("nan")
        train_ewc_raw_epoch = float("nan")
        train_ewc_weighted_epoch = float("nan")
        train_ewc_to_wm_ratio_epoch = float("nan")
        
        # B. Train World Model
        # [ALIGN] Respect warmup: freeze WM training during warmup (same as MAC)
        warmup_iters_wm = _safe_int_cfg(
            getattr(cfg.generator_agent, "warmup_iterations", 0),
            default=0,
            name="generator_agent.warmup_iterations",
        )
        is_warmup = (iteration < warmup_iters_wm)
        if (not is_warmup) and (iteration % cfg.generator_agent.wm_train_frequency == 0):
            if paired_train_seed:
                # Collection consumes RNG only in the write arm. Reset immediately
                # before WM training so both arms use the same split and batch order.
                set_seed(int(seed) + iteration)
            if is_crafter and device.type == "cuda":
                torch.cuda.synchronize(device)
            wm_train_start = time.perf_counter()
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
                if event_audit_enabled:
                    _append_crafter_event_rows(
                        event_audit_csv_path, seed, iteration + 1, "replay_selected",
                        res_train.get("replay_inventory_event_rows", []),
                    )
                if is_crafter and protected_enabled:
                    phase_path = Path(str(cfg.attention_model.model_save_path))
                    last_path = phase_path.with_name(phase_path.stem + "_last" + phase_path.suffix)
                    best_path = phase_path.with_name(phase_path.stem + "_best" + phase_path.suffix)
                    if phase_path.is_file():
                        shutil.copy2(phase_path, last_path)
                        current_selection = float(res_train.get("best_loss", float("inf")))
                        if current_selection < global_best_selection:
                            shutil.copy2(phase_path, best_path)
                            global_best_selection = current_selection
                            global_best_iteration = iteration + 1
                            print(f"  [Checkpoint] Updated fixed-DR-validation best at iteration {global_best_iteration}.")
                if transition_replay_enabled and replay_data is not None:
                    sampled_replay_stats = replay_data.get("_replay_sampling_stats")
                protected_replay_slot_counts = dict(res_train.get("protected_replay_slot_counts", {}))
                fisher_slot_counts = dict(res_train.get("fisher_slot_counts", {}))
                old_params = res_train["old_params"]
                # train_api removes the ``train/`` prefix from Lightning
                # callback metrics, making per-iteration EWC diagnostics
                # directly traceable in the experiment summary.
                train_ewc_raw = float(res_train.get("ewc_raw", float("nan")))
                train_ewc_weighted = float(res_train.get("ewc_weighted", float("nan")))
                train_ewc_to_wm_ratio = float(res_train.get("ewc_to_wm_ratio", float("nan")))
                train_ewc_raw_epoch = float(res_train.get("ewc_raw_epoch", float("nan")))
                train_ewc_weighted_epoch = float(res_train.get("ewc_weighted_epoch", float("nan")))
                train_ewc_to_wm_ratio_epoch = float(res_train.get("ewc_to_wm_ratio_epoch", float("nan")))
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
                if is_crafter and bool(getattr(cfg, "save_wm_update_checkpoints", False)):
                    raise RuntimeError(f"Cannot snapshot DR iteration {iteration + 1}; WM checkpoint reload failed: {ckpt_path}") from e
                print(f"  [Warning] Failed to reload model: {e}")
                if isinstance(old_params, dict):
                    wm.load_state_dict(old_params)
                else:
                    wm.load_state_dict(old_params.state_dict())
            generator.sync_world_model(wm.state_dict())
            if is_crafter and bool(getattr(cfg, "save_wm_update_checkpoints", False)):
                snapshot = save_wm_update_snapshot(ckpt_path, cfg.dr_log_dir, iteration + 1)
                print(f"  [Checkpoint] Saved WM update snapshot: {snapshot}")
            if is_crafter:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                wm_train_seconds = time.perf_counter() - wm_train_start

        # Keep DR's held-out LP diagnostics aligned with MAC. DR has no PPO
        # update, but the reward components remain comparable.
        if is_minigrid:
            generator.finalize_minigrid_rewards()
        elif is_crafter:
            generator.finalize_crafter_learning_progress(apply_rewards=False)

        # C. Validation on Target Tasks (aligned with MAC: validate every iter after warmup)
        warmup_iters = _safe_int_cfg(
            getattr(cfg.generator_agent, "warmup_iterations", 0),
            default=0,
            name="generator_agent.warmup_iterations",
        )
        target_val_field_losses = {
            name: float("nan") for name in MINIGRID_VAL_LOSS_FIELDS
        }
        target_val_focal_loss = 0.0
        target_val_changed_focal_loss = 0.0
        target_val_false_set_rate = 0.0
        target_val_changed_count = 0.0
        target_val_crafter_changed_nll = 0.0
        target_val_crafter_changed_count = 0.0
        target_val_crafter_inventory_metrics = {
            name: 0.0 for name in CRAFTER_INVENTORY_VAL_METRICS
        }
        target_val_crafter_focal = {name: float("nan") for name in CRAFTER_FOCAL_VAL_METRICS}
        target_val_valid_count = 0
        if iteration >= (warmup_iters - 1):
            print(f"  [Validation] Running zero-shot test on all {val_n_phases} targets...")
            task_indices = range(val_start_idx, val_start_idx + val_n_phases)
            task_names = [f"{val_task_prefix}{v_idx}" for v_idx in task_indices]
            if is_crafter and device.type == "cuda":
                torch.cuda.synchronize(device)
            target_val_start = time.perf_counter()
            val_summary = validate_on_all_targets(
                cfg,
                wm,
                val_data_path,
                task_names,
                val_suffix,
                phase_name=f"dr_iter_{iteration+1}",
                VALID_TIMES=1,
            )
            if event_audit_enabled:
                _append_crafter_target_confusion(
                    target_confusion_csv_path, seed, iteration + 1,
                    val_summary.get("crafter_event_confusion"),
                )
            if is_crafter:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                target_val_seconds = time.perf_counter() - target_val_start

            if val_summary["valid_count"] > 0:
                target_val_valid_count = int(val_summary["valid_count"])
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
                    if is_crafter:
                        target_val_crafter_changed_nll = float(val_summary.get("changed_nll", 0.0))
                        target_val_crafter_changed_count = float(val_summary.get("changed_count", 0.0))
                        target_val_crafter_inventory_metrics = {
                            name: float(val_summary.get(name, 0.0))
                            for name in CRAFTER_INVENTORY_VAL_METRICS
                        }
                        target_val_crafter_focal = {
                            name: float(val_summary.get(name, float("nan")))
                            for name in CRAFTER_FOCAL_VAL_METRICS
                        }
                    target_val_contact_acc = 0.0
                    target_val_contact_bce = 0.0
                    if is_minigrid:
                        target_val_focal_loss = float(val_summary.get("focal_loss", 0.0))
                        target_val_changed_focal_loss = float(val_summary.get("changed_focal_loss", 0.0))
                        target_val_false_set_rate = float(val_summary.get("false_set_rate", 0.0))
                        target_val_changed_count = float(val_summary.get("changed_count", 0.0))
                        target_val_field_losses = {
                            name: float(val_summary.get(name, float("nan")))
                            for name in MINIGRID_VAL_LOSS_FIELDS
                        }
                        component_summary = " | ".join(
                            f"{name}: {value:.6f}"
                            for name, value in target_val_field_losses.items()
                        )
                        print(
                            f"    -> Results: Avg Loss = "
                            f"{target_val_avg_val_loss_wm:.6f} | "
                            f"{component_summary}"
                        )
                    else:
                        print(
                            f"    -> Results: Avg Loss = {target_val_avg_val_loss_wm:.5f} | "
                            f"Changed NLL = {target_val_crafter_changed_nll:.5f} | "
                            f"Changed Count = {target_val_crafter_changed_count:.0f}"
                        )
            else:
                target_val_avg_val_loss_wm = 0.0
                target_val_val_ce_loss = 0.0
                target_val_val_inv_loss = 0.0
                target_val_crafter_changed_nll = 0.0
                target_val_crafter_changed_count = 0.0
                target_val_crafter_inventory_metrics = {
                    name: 0.0 for name in CRAFTER_INVENTORY_VAL_METRICS
                }
                target_val_contact_acc = 0.0
                target_val_contact_bce = 0.0
        else:
            target_val_avg_val_loss_wm = 0.0
            target_val_val_ce_loss = 0.0
            target_val_val_inv_loss = 0.0
            target_val_crafter_changed_nll = 0.0
            target_val_crafter_changed_count = 0.0
            target_val_crafter_inventory_metrics = {
                name: 0.0 for name in CRAFTER_INVENTORY_VAL_METRICS
            }
            target_val_contact_acc = 0.0
            target_val_contact_bce = 0.0

        # D. Buffer Archiving
        if str(cfg.domain) == "minigrid":
            fisher_buffer.add_from_batch(new_batch)
        else:
            fisher_buffer.add_from_batch(
                new_batch,
                current_sample_ratio=cfg.attention_model.current_sample_ratio,
                fisher_buffer_elements_ratio=cfg.attention_model.fisher_buffer_elements_ratio,
            )
        print(f"  [Buffer] Archived {current_transitions} transitions. Buffer Size: {len(fisher_buffer)}")
        buffer_transition_stats = (
            fisher_buffer.transition_replay_stats()
            if transition_replay_enabled and len(fisher_buffer) else None
        )
        if is_minigrid:
            replay_data_after = fisher_buffer.export_dict() if len(fisher_buffer) else None
            replay_changed_fraction, _ = minigrid_changed_fraction(replay_data_after)
            _, batch_changed_count = minigrid_changed_fraction(new_batch)
            mg_metrics = getattr(generator, "last_minigrid_metrics", {})
            row_data = {
                "Seed": seed, "Iter": iteration + 1, "Gen_Mean_Reward": 0.0, "Gen_Loss": 0.0,
                "Gen_Entropy": 0.0, "Gen_Div_Reward": gen_div_reward,
                "gen_val_avg_val_loss_wm": gen_val_avg_val_loss_wm,
                "target_val_avg_val_loss_wm": target_val_avg_val_loss_wm,
                "target_val_valid_count": target_val_valid_count,
                "target_val_focal_loss": target_val_focal_loss,
                "target_val_changed_focal_loss": target_val_changed_focal_loss,
                "target_val_false_set_rate": target_val_false_set_rate,
                "target_val_changed_count": target_val_changed_count,
                "New_Data_Size": current_transitions, "Buffer_Size": len(fisher_buffer),
                "Solvable_Count": solvable_count, "Avg_Path_Len": avg_path_len,
                "Replay_Changed_Fraction": replay_changed_fraction,
                "Batch_Changed_Count": batch_changed_count,
                "Map_Novelty": mg_metrics.get("Map_Novelty", 0.0),
                "Combination_Novelty": mg_metrics.get("Combination_Novelty", 0.0),
                "Random_Feature_Novelty": mg_metrics.get("Random_Feature_Novelty", 0.0),
                "Pre_Changed_Focal_Loss": mg_metrics.get("Pre_Changed_Focal_Loss", 0.0),
                "Post_Changed_Focal_Loss": mg_metrics.get("Post_Changed_Focal_Loss", 0.0),
                "Learning_Progress": mg_metrics.get("Learning_Progress", 0.0),
                "Difficulty_Rank": mg_metrics.get("Difficulty_Rank", 0.0),
                "Learning_Progress_Rank": mg_metrics.get("Learning_Progress_Rank", 0.0),
                "Novelty_Rank": mg_metrics.get("Novelty_Rank", 0.0),
                "Batch_Nearest_Hamming": mg_metrics.get("Batch_Nearest_Hamming", 0.0),
                "Archive_Nearest_Hamming": mg_metrics.get("Archive_Nearest_Hamming", 0.0),
                "Novelty_Distance_Std": mg_metrics.get("Novelty_Distance_Std", 0.0),
                "Latent_Batch_LogDet": mg_metrics.get("Latent_Batch_LogDet", 0.0),
                "Mean_Object_Pair_Distance": mg_metrics.get("Mean_Object_Pair_Distance", 0.0),
                "Mean_Nearest_Object_Distance": mg_metrics.get("Mean_Nearest_Object_Distance", 0.0),
                "Selected_Edit_Pair_Distance": mg_metrics.get("Selected_Edit_Pair_Distance", 0.0),
                "Mean_Edit_Rate": mg_metrics.get("Mean_Edit_Rate", 0.0),
                "Unique_Goal_Positions": mg_metrics.get("Unique_Goal_Positions", 0),
                "Reward_Learning_Progress": mg_metrics.get("Reward_Learning_Progress", 0.0),
                "Reward_Combination_Novelty": mg_metrics.get("Reward_Combination_Novelty", 0.0),
                "Reward_Random_Feature_Novelty": mg_metrics.get("Reward_Random_Feature_Novelty", 0.0),
                "Final_Generator_Reward": mg_metrics.get("Final_Generator_Reward", 0.0),
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
                "Seed": seed, "Iter": iteration + 1,
                "New_Data_Size": current_transitions,
                "Cumulative_Transitions": cumulative_transitions,
                "Buffer_Size": len(fisher_buffer),
                "target_val_valid_count": target_val_valid_count,
                "target_val_sample_count": target_val_valid_count * int(
                    getattr(cfg.attention_model, "target_validation_max_samples", 0)
                ),
                "target_val_avg_val_loss_wm": target_val_avg_val_loss_wm,
                "target_val_inventory_effect_false_positive_rate": target_val_crafter_inventory_metrics.get("inventory_effect_false_positive_rate", float("nan")),
                "target_val_inventory_effect_change_recall": target_val_crafter_inventory_metrics.get("inventory_effect_change_recall", float("nan")),
                "target_val_inventory_effect_change_precision": target_val_crafter_inventory_metrics.get("inventory_effect_change_precision", float("nan")),
                "target_val_inventory_effect_row_exact": target_val_crafter_inventory_metrics.get("inventory_effect_row_exact", float("nan")),
                "paired_batch_hash": paired_batch_hash,
                "target_val_changed_focal_loss": target_val_crafter_focal["changed_focal_loss"],
                "target_val_layout_changed_focal_loss": target_val_crafter_focal["layout_changed_focal_loss"],
                "target_val_layout_false_set_rate": target_val_crafter_focal["layout_false_set_rate"],
                "target_val_layout_changed_count": target_val_crafter_focal["layout_changed_count"],
                "target_val_inventory_changed_focal_loss": target_val_crafter_focal["inventory_changed_focal_loss"],
                "target_val_inventory_false_set_rate": target_val_crafter_focal["inventory_false_set_rate"],
                "target_val_inventory_changed_count": target_val_crafter_focal["inventory_changed_count"],
                "target_val_joint_accuracy": target_val_crafter_focal["joint_accuracy"],
                "target_val_position_accuracy": target_val_crafter_focal["position_accuracy"],
                "target_val_direction_accuracy": target_val_crafter_focal["direction_accuracy"],
                "Gen_Mean_Reward": 0.0, "Gen_Loss": 0.0, "Gen_Entropy": 0.0,
                "Gen_Div_Reward": gen_div_reward, "Inv_Change_Ratio": inv_change_ratio,
                "Pre_Changed_Focal_Loss": getattr(generator, "last_crafter_metrics", {}).get("Pre_Changed_Focal_Loss", float("nan")),
                "Post_Changed_Focal_Loss": getattr(generator, "last_crafter_metrics", {}).get("Post_Changed_Focal_Loss", float("nan")),
                "Learning_Progress": getattr(generator, "last_crafter_metrics", {}).get("Learning_Progress", float("nan")),
                **{
                    key: getattr(generator, "last_crafter_metrics", {}).get(
                        key,
                        float("nan") if getattr(generator, "skip_crafter_learning_progress", False) else 0.0,
                    )
                    for key in (
                        "Inventory_KEEP_Ratio",
                        *(f"Inventory_Stage_{stage}_{metric}" for stage in range(5) for metric in ("Count", "Mean_LP", "Reward_Std")),
                    )
                },
                "Solvable_Count": solvable_count, "Avg_Path_Len": avg_path_len,
            }
            if quick_target_loss_csv:
                row_data.update({
                    "target_val_terrain_loss": target_val_val_ce_loss,
                    "target_val_inventory_loss": target_val_val_inv_loss,
                    "target_val_changed_nll": target_val_crafter_changed_nll,
                })
                for loss_name, changed_count in (
                    ("target_val_changed_focal_loss", target_val_crafter_changed_count),
                    ("target_val_layout_changed_focal_loss", target_val_crafter_focal["layout_changed_count"]),
                    ("target_val_inventory_changed_focal_loss", target_val_crafter_focal["inventory_changed_count"]),
                ):
                    if changed_count == 0 and not np.isfinite(row_data[loss_name]):
                        row_data[loss_name] = 0.0
            if quick_ewc_diagnostics_csv:
                row_data.update({
                    "train_ewc_raw_epoch": train_ewc_raw_epoch,
                    "train_ewc_weighted_epoch": train_ewc_weighted_epoch,
                    "train_ewc_to_wm_ratio_epoch": train_ewc_to_wm_ratio_epoch,
                })
            if generator.skip_crafter_learning_progress:
                skipped_lp_columns = (
                    "Pre_Changed_Focal_Loss", "Post_Changed_Focal_Loss",
                    "Learning_Progress", "Inventory_KEEP_Ratio",
                    *(f"Inventory_Stage_{stage}_{metric}"
                      for stage in range(5)
                      for metric in ("Count", "Mean_LP", "Reward_Std")),
                )
                row_data.update({name: 0.0 for name in skipped_lp_columns})

        pd.DataFrame([row_data], columns=csv_columns).to_csv(
            summary_csv_path,
            index=False,
            mode="a",
            header=not file_exists,
        )
        file_exists = True
        if transition_replay_enabled and transition_stats_csv_path is not None:
            detail_rows = []
            for source, stats in (
                ("current", current_transition_stats),
                ("buffer", buffer_transition_stats),
                ("sampled_replay", sampled_replay_stats),
            ):
                for transition_type, values in (stats or {}).items():
                    for action, count in values["actions"].items():
                        detail_rows.append({
                            "seed": seed, "iteration": iteration + 1, "source": source,
                            "transition_type": transition_type, "action": action,
                            "changed_slot": None, "count": count,
                        })
                    for slot, count in values.get("changed_slots", {}).items():
                        detail_rows.append({
                            "seed": seed, "iteration": iteration + 1, "source": source,
                            "transition_type": transition_type, "action": "all",
                            "changed_slot": int(slot), "count": count,
                        })
            # Slot-only sources deliberately have no transition-type/action
            # rows; the legacy rows above stay intact for those dimensions.
            train_slot_counts = _sum_slot_counts(
                _slot_counts(current_transition_stats), _slot_counts(sampled_replay_stats)
            )
            for source, counts in (
                ("protected_replay", protected_replay_slot_counts),
                ("train", train_slot_counts),
                ("fisher", fisher_slot_counts),
                ("validation", validation_slot_counts),
            ):
                for slot, count in (counts or {}).items():
                    detail_rows.append({
                        "seed": seed, "iteration": iteration + 1, "source": source,
                        "transition_type": "inventory_change", "action": "all",
                        "changed_slot": int(slot), "count": int(count),
                    })
            if detail_rows:
                pd.DataFrame(detail_rows).to_csv(
                    transition_stats_csv_path, mode="a",
                    header=not transition_stats_csv_path.exists(), index=False,
                )
        if is_crafter:
            timing = generator.crafter_timing
            print(
                "  [Timing] "
                f"collection={timing['collection']:.3f}s "
                f"lp_pre={timing['lp_pre']:.3f}s "
                f"wm_train={wm_train_seconds:.3f}s "
                f"lp_post={timing['lp_post']:.3f}s "
                f"target_val={target_val_seconds:.3f}s"
            )
        torch.cuda.empty_cache()

        # E. Cleanup Temporary Data
        data_save_dir = Path(getattr(cfg.env.collect, "data_folder", str(TRAINER_PATH / "data")))
        temp_files = glob.glob(str(data_save_dir / f"UED_Dual_iter{iteration}_b*.npz"))
        if temp_files:
            print(f"  [Cleanup] Deleting {len(temp_files)} temporary files for Iteration {iteration}...")
            for f in temp_files:
                try: os.remove(f)
                except: pass

        if is_crafter and bool(getattr(cfg, "save_iteration_state", True)):
            save_state(resume_state_path, {
                "version": 1, "kind": "crafter_dr", "config_digest": config_digest(cfg),
                "completed_iteration": iteration + 1,
                "cumulative_transitions": cumulative_transitions,
                "wm": wm.state_dict(), "old_params": old_params, "fisher": fisher,
                "replay": replay_state(fisher_buffer),
                "generator": generator_state(generator, include_policy=False),
                "global_best_selection": global_best_selection,
                "global_best_iteration": global_best_iteration,
                "rng": rng_state(),
            })

    print(f"\n[DR DONE] Log: {summary_csv_path}")

if __name__ == "__main__":
    run_dr_baseline_experiment()
