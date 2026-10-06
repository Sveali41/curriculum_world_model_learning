import sys
import os
import json
import csv
import subprocess
ROOT_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WM_ROOT = os.path.join(ROOT_DIR, "wm")
# Keep trainer results in the outer workspace even if a shell inherited a
# PROJECT_ROOT value pointing at the nested WM tree.
os.environ["TRAINER_ROOT"] = ROOT_DIR
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
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import torch
import glob
import torch
import numpy as np
import copy
import math
import time
import gc

from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.continue_learning.reservoir_buffer import ReservoirReplayBuffer
from modelBased.common.artifacts import align_world_model_artifact_path
from modelBased.exploration.minigrid_corpus import MiniGridCorpusWriter

from generator.generator_interface import GeneratorInterface
from trainer.common.run_resume import (
    config_digest, generator_state, guard_fresh_results, load_state, replay_state, restore_generator,
    restore_replay, restore_rng, rng_state, save_state, validate_sidecars,
)
from trainer.common.wm_snapshots import save_wm_update_snapshot
from trainer.common.paths import RESULTS_ROOT
from trainer.common.utils import (
    MINIGRID_VAL_LOSS_FIELDS,
    CRAFTER_FOCAL_VAL_METRICS,
    CRAFTER_INVENTORY_VAL_METRICS,
    set_seed,
    validate_on_target_task,
    save_validation_csv,
    convert_trajectories_to_batch,
    minigrid_changed_fraction,
)
from modelBased.world_model.crafter_event_audit import (
    inventory_event_episode_age_rows, inventory_event_rows,
)


CRAFTER_INVENTORY_EVENT_AUDIT_HEADER = (
    "Seed", "Iter", "Source", "Event_ID", "Delta_Vector", "Count",
    "Changed_Slot_Count", "Changed_Slots",
)
CRAFTER_TARGET_EVENT_CONFUSION_HEADER = (
    "Seed", "Iter", "Target", "True_Event_ID", "True_Delta_Vector",
    "True_Count", "Predicted_Exact_Count", "Predicted_KEEP_Count",
    "Predicted_Other_Count", "True_KEEP_False_Positive_Count",
    "True_Change_Margin_Count", "True_Change_Margin_Mean",
)
CRAFTER_TARGET_EVENT_SUMMARY_HEADER = (
    "Seed", "Iter", "Target", "Sample_Count", "True_KEEP_Count",
    "True_KEEP_False_Positive_Count", "Slot_TP", "Slot_FN", "Slot_FP",
    "Slot_KEEP_Count", "Unknown_Event_Count",
)


def _append_inventory_event_audit(csv_path: Path, seed: int, iteration: int, source: str, rows):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(CRAFTER_INVENTORY_EVENT_AUDIT_HEADER)
        writer.writerows((
            seed, iteration, source, row["event_id"], row["delta_vector"],
            row["count"], row["changed_slot_count"], row["changed_slots"],
        ) for row in rows)


def _append_episode_age_event_audit(csv_path: Path, seed: int, iteration: int, rollouts):
    header = ("Seed", "Iter", "Rollout", "Episode_Age_Bin", "Bin_Transition_Count",
              "Episodes_With_Transitions_In_Bin", "Event_ID", "Delta_Vector",
              "Event_Count", "Rate_Per_1000", "Changed_Slot_Count", "Changed_Slots")
    rows = []
    for rollout_index, rollout in enumerate(rollouts):
        if not all(rollout.get(key) is not None for key in ("inv", "inv_next", "done")):
            raise ValueError(f"Crafter rollout {rollout_index} lacks inventory/done fields for episode-age audit")
        rows.extend((rollout_index, row) for row in inventory_event_episode_age_rows(
            rollout["inv"], rollout["inv_next"], rollout["done"]
        ))
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(header)
        for rollout_index, row in rows:
            writer.writerow((seed, iteration, rollout_index, row["episode_age_bin"],
                             row["bin_transition_count"], row["episodes_with_transitions_in_bin"],
                             row["event_id"], row["delta_vector"], row["event_count"],
                             f"{row['rate_per_1000']:.8f}", row["changed_slot_count"], row["changed_slots"]))


def _append_target_event_confusion(csv_path: Path, summary_csv_path: Path, seed: int, iteration: int, target: str, payload):
    if not payload:
        return
    if isinstance(payload, dict):
        rows = []
        keep_count = int(payload.get("true_keep_count", 0))
        keep_false_positive = int(payload.get("true_keep_false_positive", 0))
        rows.append({
            "true_event_id": 0,
            "true_delta_vector": json.dumps([0] * 12, separators=(",", ":")),
            "true_count": keep_count,
            "predicted_exact_count": keep_count - keep_false_positive,
            "predicted_keep_count": keep_count - keep_false_positive,
            "predicted_other_count": keep_false_positive,
            "true_keep_false_positive_count": keep_false_positive,
            "true_change_margin_count": 0,
            "true_change_margin_mean": float("nan"),
        })
        for event in payload.get("events", []):
            margin_count = int(event.get("margin_count", 0))
            event_id = -1 if event.get("event_id") is None else int(event["event_id"])
            rows.append({
                "true_event_id": event_id,
                "true_delta_vector": json.dumps(event.get("delta", ()), separators=(",", ":")),
                "true_count": int(event.get("count", 0)),
                "predicted_exact_count": int(event.get("exact", 0)),
                "predicted_keep_count": int(event.get("predicted_keep", 0)),
                "predicted_other_count": int(event.get("predicted_other", 0)),
                "true_keep_false_positive_count": (
                    int(payload.get("true_keep_false_positive", 0))
                    if event_id == 0 else 0
                ),
                "true_change_margin_count": margin_count,
                "true_change_margin_mean": (
                    float(event.get("margin_sum", 0.0)) / margin_count
                    if margin_count else float("nan")
                ),
            })
    else:
        rows = payload
    if isinstance(payload, dict):
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(CRAFTER_TARGET_EVENT_SUMMARY_HEADER)
            slot_tp = int(payload.get("slot_true_positive", 0))
            slot_fn = int(payload.get("slot_false_negative", 0))
            slot_fp = int(payload.get("slot_false_positive", 0))
            writer.writerow((
                seed, iteration, target, int(payload.get("sample_count", 0)),
                int(payload.get("true_keep_count", 0)),
                int(payload.get("true_keep_false_positive", 0)),
                slot_tp, slot_fn, slot_fp, int(payload.get("slot_keep_count", 0)),
                int(payload.get("unknown_event_count", 0)),
            ))
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(CRAFTER_TARGET_EVENT_CONFUSION_HEADER)
        for row in rows:
            writer.writerow((
                seed, iteration, target, row["true_event_id"], row["true_delta_vector"],
                row["true_count"], row["predicted_exact_count"], row["predicted_keep_count"],
                row["predicted_other_count"], row.get("true_keep_false_positive_count", 0),
                row.get("true_change_margin_count", 0), row.get("true_change_margin_mean", float("nan")),
            ))



CRAFTER_STAGE_PROBE_CSV_HEADER = [
    "Seed", "Iter", "Map_Index", "Stage_Token", "Environment_Valid",
    "Probe_Valid", "Layout_Learning_Progress", "Inventory_Learning_Progress",
    "Layout_LP_Scale", "Inventory_LP_Scale", "Normalized_Layout_LP",
    "Normalized_Inventory_LP", "Layout_Reward_Layout_LP",
    "Layout_Reward_Inventory_LP", "Stage_Reward_Layout_LP",
    "Stage_Reward_Inventory_LP", "Layout_Reward_Layout_LP_Share",
    "Layout_Reward_Inventory_LP_Share", "Layout_Reward_Novelty_Share",
    "Stage_Reward_Layout_LP_Share", "Stage_Reward_Inventory_LP_Share",
    "Stage_Reward_Novelty_Share", "Inventory_Changed_Slots",
    "Inventory_Changed_Count_Pre", "Stage_Reward",
]

def _append_crafter_stage_probe_rows(csv_path: Path, seed: int, iteration: int, probe_rows):
    """Append per-generated-environment inventory LP diagnostics."""
    import csv

    if not probe_rows:
        return 0
    with Path(csv_path).open("a", newline="") as probe_file:
        writer = csv.writer(probe_file)
        for row in probe_rows:
            writer.writerow([
                seed, iteration, row.get("map_index", -1),
                row.get("stage_token", -1),
                int(bool(row.get("environment_valid", False))),
                int(bool(row.get("valid", False))),
                row.get("layout_learning_progress", float("nan")),
                row.get("inventory_learning_progress", float("nan")),
                row.get("layout_lp_scale", float("nan")),
                row.get("inventory_lp_scale", float("nan")),
                row.get("normalized_layout_lp", float("nan")),
                row.get("normalized_inventory_lp", float("nan")),
                row.get("layout_reward_layout_lp", float("nan")),
                row.get("layout_reward_inventory_lp", float("nan")),
                row.get("stage_reward_layout_lp", float("nan")),
                row.get("stage_reward_inventory_lp", float("nan")),
                row.get("layout_reward_layout_lp_share", float("nan")),
                row.get("layout_reward_inventory_lp_share", float("nan")),
                row.get("layout_reward_novelty_share", float("nan")),
                row.get("stage_reward_layout_lp_share", float("nan")),
                row.get("stage_reward_inventory_lp_share", float("nan")),
                row.get("stage_reward_novelty_share", float("nan")),
                row.get("inventory_changed_slots", 0),
                row.get("inventory_changed_count_pre", 0),
                row.get("reward_stage", float("nan")),
            ])
    return len(probe_rows)


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


def _validate_existing_result_rows(csv_path: Path, expected_columns, seed):
    """Validate rows before appending a new seed to a shared result CSV.

    MAC runs are fresh per seed, so an existing row for the same seed almost
    always means that the run would be duplicated.  Other seeds are allowed to
    coexist in the same file, while duplicate ``(Seed, Iter)`` keys are
    rejected for every seed.
    """
    import csv

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    seen_keys = set()
    existing_seed_rows = 0
    expected_header = list(expected_columns)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"Existing MAC CSV has an incompatible schema: {csv_path}"
            )
        for row_number, row in enumerate(reader, start=2):
            if not row or all(value in (None, "") for value in row.values()):
                continue
            try:
                row_seed = int(row["Seed"])
                row_iter = int(row["Iter"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Seed/Iter at {csv_path}:{row_number}"
                ) from exc
            key = (row_seed, row_iter)
            if key in seen_keys:
                raise ValueError(
                    f"Duplicate (Seed, Iter)={key} in existing MAC CSV: {csv_path}"
                )
            seen_keys.add(key)
            if row_seed == int(seed):
                existing_seed_rows += 1

    if existing_seed_rows:
        raise FileExistsError(
            f"Refusing to append MAC seed {int(seed)}: {csv_path} already contains "
            f"{existing_seed_rows} row(s) for this seed. Choose a new seed or "
            "remove/backup the previous run."
        )


def _write_run_manifest(path: Path, cfg: DictConfig):
    """Persist run provenance so historical MAC curves remain attributable."""
    def git_value(args, cwd):
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, check=True, text=True,
                capture_output=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    manifest = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "outer_git_commit": git_value(["rev-parse", "HEAD"], ROOT_DIR),
        "outer_git_status": git_value(["status", "--porcelain"], ROOT_DIR),
        "wm_git_commit": git_value(["rev-parse", "HEAD"], WM_ROOT),
        "wm_git_status": git_value(["status", "--porcelain"], WM_ROOT),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

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
    diagnostics_cfg = getattr(cfg, "diagnostics", None)
    strict_results = bool(getattr(diagnostics_cfg, "strict_results", False))
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
    resume_training = env_type == "crafter" and bool(getattr(cfg, "resume_training", False))
    if resume_training and bool(getattr(cfg, "force_fresh_start", False)):
        raise ValueError("resume_training=true requires force_fresh_start=false")
    transition_replay_cfg = getattr(cfg.attention_model, "crafter_transition_replay", None)
    if env_type == "crafter" and bool(getattr(transition_replay_cfg, "enabled", False)):
        raise ValueError(
            "Crafter MAC uses the ordinary FisherReplayBuffer; set "
            "attention_model.crafter_transition_replay.enabled=false."
        )
    mask_suffix = f"_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"

    if env_type == "minigrid":
        summary_csv_path = log_dir / (
            f"minigrid_ued_results_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"
            f"_focal_reservoir_sa{ablation_suffix}.csv"
        )
    elif env_type == "crafter":
        summary_csv_path = log_dir / (
            f"mac_crafter_lp_layout_stage_novelty_results{ablation_suffix}{metric_suffix}.csv"
        )
    elif env_type == "bipedalwalker":
        summary_csv_path = log_dir / (
            f"bipedalwalker_ued_lp_g1_results{mask_suffix}{ablation_suffix}{metric_suffix}.csv"
        )
    else:
        summary_csv_path = log_dir / f"{env_type}_ued_results{mask_suffix}{ablation_suffix}{metric_suffix}.csv"
    if ablation_suffix or metric_suffix or env_type != "crafter":
        print(f"[Log] CSV Path Adjusted: {summary_csv_path}")

    # Keep one provenance file per seed; the summary CSV and map sidecar are
    # intentionally shared across seeds.
    run_manifest_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.seed{int(seed)}.manifest.json"
    )
    map_diagnostics_path = summary_csv_path.with_suffix(".maps.jsonl")
    resume_state_path = log_dir / f"crafter_mac_seed{seed}.resume.pt"
    stage_probe_csv_path = None
    stage_probe_csv_header = CRAFTER_STAGE_PROBE_CSV_HEADER
    if (env_type == "crafter" and str(getattr(cfg.generator_agent, "crafter_credit_mode", "joint"))
            in {"split", "split_balanced_global"}):
        stage_probe_csv_path = summary_csv_path.with_name(
            f"{summary_csv_path.stem}.stage_probe_rows.csv"
        )
    event_audit_cfg = getattr(cfg, "crafter_inventory_event_audit", None)
    event_audit_enabled = bool(
        event_audit_cfg.get("enabled", False) if isinstance(event_audit_cfg, dict)
        else getattr(event_audit_cfg, "enabled", False)
    ) and env_type == "crafter"
    event_audit_csv_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.inventory_events.csv"
    ) if event_audit_enabled else None
    episode_age_audit_csv_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.inventory_events_by_episode_age.csv"
    ) if event_audit_enabled else None
    target_event_confusion_csv_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.target_event_confusion.csv"
    ) if event_audit_enabled else None
    target_event_summary_csv_path = summary_csv_path.with_name(
        f"{summary_csv_path.stem}.target_event_summary.csv"
    ) if event_audit_enabled else None
    if event_audit_enabled and not bool(
        getattr(cfg.attention_model, "crafter_event_confusion_enabled", False)
    ):
        raise ValueError(
            "Crafter inventory event audit requires "
            "attention_model.crafter_event_confusion_enabled=true"
        )

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
    is_crafter = (env_type == "crafter")
    if is_bipedal:
        csv_header = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_contact_acc", "gen_val_contact_bce", "gen_val_avg_val_loss_wm",
            "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
            "target_val_contact_changed_loss",
            "Pre_WM_Loss", "Post_WM_Loss", "Learning_Progress", "LP_Probe_Count",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Avg_Path_Len",
        ]
    elif is_minigrid:
        csv_header = [
            "Seed", "Iter", "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "gen_val_avg_val_loss_wm", "target_val_avg_val_loss_wm",
            "target_val_valid_count",
            "target_val_focal_loss", "target_val_changed_focal_loss",
            "target_val_false_set_rate", "target_val_changed_count",
            "Learning_Progress", "Difficulty_Rank", "Learning_Progress_Rank",
            "New_Data_Size", "Buffer_Size", "Solvable_Count", "Solvable_Rate", "Avg_Path_Len",
            "Replay_Changed_Fraction", "Batch_Changed_Count",
            "Map_Novelty", "Combination_Novelty", "Random_Feature_Novelty",
            "Pre_Changed_Focal_Loss", "Post_Changed_Focal_Loss", "Novelty_Rank", "Batch_Nearest_Hamming",
            "Archive_Nearest_Hamming", "Novelty_Distance_Std", "Latent_Batch_LogDet",
            "Mean_Object_Pair_Distance", "Mean_Nearest_Object_Distance",
            "Selected_Edit_Pair_Distance", "Mean_Edit_Rate", "Unique_Goal_Positions",
            "Reward_Learning_Progress", "Reward_Combination_Novelty",
            "Reward_Random_Feature_Novelty", "Final_Generator_Reward",
            "Explorer_Coverage_Rate", "Explorer_Unique_Positions",
            "Explorer_Walkable_Cells",
            "LP_Split_Spearman", "PPO_Updated", "PPO_Num_Samples",
            "PPO_Policy_Loss", "PPO_Value_Loss", "PPO_Approx_KL",
            "PPO_Clip_Fraction", "PPO_Ratio_Mean", "PPO_Ratio_Min",
            "PPO_Ratio_Max", "PPO_Reward_Std", "PPO_Advantage_Std",
            "PPO_Initial_Logprob_Error", "PPO_Grad_Norm", "PPO_Param_Delta",
        ]
    else:
        csv_header = [
            "Seed", "Iter",
            "Gen_Mean_Reward", "Gen_Loss", "Gen_Entropy", "Gen_Div_Reward",
            "Novelty_Total_Mean", "Novelty_Total_Std",
            "Pre_Changed_Focal_Loss", "Post_Changed_Focal_Loss", "Learning_Progress",
            "Pre_Layout_Changed_Focal_Loss", "Post_Layout_Changed_Focal_Loss", "Layout_Learning_Progress", "Layout_Paired_Probe_Count",
            "Pre_Inventory_Changed_Focal_Loss", "Post_Inventory_Changed_Focal_Loss", "Inventory_Learning_Progress", "Inventory_Paired_Probe_Count",
            "Reward_LP_Mean", "Reward_Novelty_Mean",
            "Layout_LP_Scale", "Inventory_LP_Scale",
            "Normalized_Layout_LP_Abs_Mean", "Normalized_Inventory_LP_Abs_Mean",
            "Layout_Reward_Layout_LP_Abs_Mean", "Layout_Reward_Inventory_LP_Abs_Mean",
            "Layout_Reward_Layout_LP_Share", "Layout_Reward_Inventory_LP_Share",
            "Layout_Reward_Novelty_Abs_Mean", "Layout_Reward_Novelty_Share",
            "Stage_Reward_Layout_LP_Abs_Mean", "Stage_Reward_Inventory_LP_Abs_Mean",
            "Stage_Reward_Layout_LP_Share", "Stage_Reward_Inventory_LP_Share",
            "Stage_Reward_Novelty_Abs_Mean", "Stage_Reward_Novelty_Share",
            "Final_Reward_Mean", "Final_Reward_Std", "Valid_Probe_Count", "Paired_Probe_Count",
            "PPO_Updated", "PPO_Num_Samples", "PPO_Policy_Loss", "PPO_Value_Loss",
            "PPO_Approx_KL", "PPO_Clip_Fraction", "PPO_Ratio_Mean", "PPO_Ratio_Min",
            "PPO_Ratio_Max", "PPO_Reward_Std", "PPO_Advantage_Std",
            "PPO_Initial_Logprob_Error", "PPO_Grad_Norm", "PPO_Param_Delta",
            "Layout_Reward_Mean", "Layout_Reward_Std", "Stage_Reward_Mean", "Stage_Reward_Std",
            "Stage_No_Inventory_Event_Count", "PPO_Layout_Policy_Loss", "PPO_Stage_Policy_Loss",
            "PPO_Layout_Initial_Logprob_Error", "PPO_Stage_Initial_Logprob_Error",
            "PPO_Shared_Actor_Gradient_Cosine", "PPO_Shared_Actor_Gradient_Conflict",
            "PPO_Shared_Actor_Gradient_Conflict_Ratio",
            "PPO_Shared_Layout_Gradient_Norm", "PPO_Shared_Stage_Gradient_Norm",
            "PPO_Stage_Head_Policy_Gradient_Norm", "PPO_Stage_Head_Param_Delta",
            "PPO_Stage_Policy_KL_Pre_Post", "PPO_Stage_Entropy_Pre",
            "PPO_Stage_Entropy_Post",
            "Inventory_KEEP_Ratio", "Inventory_Stage_0_Count", "Inventory_Stage_0_Mean_LP", "Inventory_Stage_0_Reward_Std", "Inventory_Stage_1_Count", "Inventory_Stage_1_Mean_LP", "Inventory_Stage_1_Reward_Std", "Inventory_Stage_2_Count", "Inventory_Stage_2_Mean_LP", "Inventory_Stage_2_Reward_Std", "Inventory_Stage_3_Count", "Inventory_Stage_3_Mean_LP", "Inventory_Stage_3_Reward_Std", "Inventory_Stage_4_Count", "Inventory_Stage_4_Mean_LP", "Inventory_Stage_4_Reward_Std",
            "Inv_Changed_Slots_Mean", "Inv_Change_Ratio", "Solvable_Count", "Avg_Path_Len",
            "New_Data_Size", "Cumulative_Transitions", "Buffer_Size",
            "target_val_valid_count", "target_val_sample_count", "target_val_avg_val_loss_wm",
            "target_val_layout_changed_focal_loss", "target_val_inventory_changed_focal_loss",
            "target_val_changed_focal_loss",
            "target_val_layout_false_set_rate", "target_val_layout_changed_count",
            "target_val_inventory_false_set_rate", "target_val_inventory_changed_count",
            "target_val_inventory_effect_false_positive_rate",
            "target_val_inventory_effect_change_recall",
            "target_val_inventory_effect_change_precision",
            "target_val_inventory_effect_row_exact",
            "target_val_joint_accuracy", "target_val_position_accuracy", "target_val_direction_accuracy",
        ]

    if env_type == "crafter" and not resume_training and bool(getattr(cfg, "save_iteration_state", True)):
        guard_fresh_results(summary_csv_path, resume_state_path, seed)
    file_exists = (
        summary_csv_path.is_file()
        if resume_training else _ensure_csv_header_compatible(summary_csv_path, csv_header)
    )
    if strict_results and not resume_training:
        _validate_existing_result_rows(summary_csv_path, csv_header, seed)
    if not resume_training:
        _write_run_manifest(run_manifest_path, cfg)
    if not file_exists:
        with open(summary_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            print(f"[Logger] Experiment summary initialized with {len(csv_header)} columns at {summary_csv_path}")
    else:
        print(f"[Logger] Reusing existing summary CSV: {summary_csv_path}")
    print(f"[Logger] Experiment summary will be saved to {summary_csv_path}")
    if stage_probe_csv_path is not None:
        if stage_probe_csv_path.exists():
            with stage_probe_csv_path.open(newline="") as probe_file:
                existing_header = next(csv.reader(probe_file), None)
            if existing_header != stage_probe_csv_header:
                raise ValueError(
                    f"Stage probe CSV schema mismatch: {stage_probe_csv_path}"
                )
        else:
            with stage_probe_csv_path.open("w", newline="") as probe_file:
                csv.writer(probe_file).writerow(stage_probe_csv_header)
        print(f"[Logger] Stage probe rows will be saved to {stage_probe_csv_path}")
    
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
            backend = str(getattr(domain_cfg.rmax_like, "backend", "ppo")).lower()
            checkpoint_paths.append(
                getattr(
                    domain_cfg.rmax_like,
                    "dqn_checkpoint_path" if backend == "dqn" else "checkpoint_path",
                    domain_cfg.rmax_like.checkpoint_path,
                )
            )
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
    
    if resume_training:
        pass  # The full iteration state is loaded after all state holders exist.
    elif os.path.exists(ckpt_path):
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
    cumulative_transitions = 0
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
    start_iteration = 0
    if resume_training:
        state = load_state(resume_state_path, cfg, "crafter_mac", summary_csv_path, csv_header)
        validate_sidecars(
            tuple(path for path in (stage_probe_csv_path, event_audit_csv_path,
                 episode_age_audit_csv_path, target_event_confusion_csv_path,
                 target_event_summary_csv_path) if path is not None),
            (map_diagnostics_path,), seed, state["completed_iteration"],
        )
        wm_instance.load_state_dict(state["wm"])
        old_params, fisher = state["old_params"], state["fisher"]
        restore_replay(fisher_buffer, state["replay"])
        restore_generator(gen_interface, state["generator"], include_policy=True)
        cumulative_transitions = state["cumulative_transitions"]
        warmup_cleanup_done = state["warmup_cleanup_done"]
        start_iteration = state["completed_iteration"]
        restore_rng(state["rng"])
        print(f"[Resume] Crafter MAC completed iteration {start_iteration}; continuing at {start_iteration + 1}")

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
    for iteration in range(start_iteration, total_iterations):
        print(
            f"\n=== Iteration {iteration + 1}/{total_iterations} ==="
        )
        inv_change_ratio = 0.0
        wm_train_seconds = 0.0
        target_val_seconds = 0.0
        train_res = {}
        current_inventory_event_rows = []
        current_episode_age_rollouts = []

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
        if event_audit_enabled:
            current_episode_age_rollouts = valid_trajectories
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
        # An empty trajectory list currently returns an empty legacy-shaped
        # batch with ``info=None``.  Do not send that placeholder through the
        # MiniGrid DataModule: it is not a legacy dataset and has no inventory
        # transitions to supervise.
        if new_batch is not None and len(new_batch.get("obs", [])) == 0:
            print(
                "[World Model] No valid transitions collected; skipping this "
                "iteration's WM update."
            )
            new_batch = None
        new_data_size = 0
        buffer_input = None  # Init for later use

        if new_batch is not None:
            new_data_size = len(new_batch['obs'])
            if event_audit_enabled and new_batch.get("inv") is not None and new_batch.get("inv_next") is not None:
                current_inventory_event_rows = inventory_event_rows(
                    new_batch["inv"], new_batch["inv_next"]
                )
            cumulative_transitions += int(new_data_size)
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
                        inv_change_ratio = float((delta[:, 4:16] > 1e-6).mean())
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
            if bool(getattr(cfg, "wm_train_seed_per_update", False)):
                set_seed(int(seed) + iteration - warmup_iterations)
            if env_type == "crafter" and device.type == "cuda":
                torch.cuda.synchronize(device)
            wm_train_start = time.perf_counter()
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
                if env_type == "crafter" and bool(getattr(cfg, "save_wm_update_checkpoints", False)):
                    raise RuntimeError(f"Cannot snapshot MAC iteration {iteration + 1}; WM checkpoint reload failed: {ckpt_path}") from e
                print(f"[Warning] Failed to reload model: {e}. Using potentially dirty instance.")
                if isinstance(old_params, dict):
                     wm_instance.load_state_dict(old_params)
                else:
                     wm_instance.load_state_dict(old_params.state_dict())

            # Resynchronize the generator with the updated world model.
            gen_interface.sync_world_model(wm_instance.state_dict())
            if env_type == "crafter" and bool(getattr(cfg, "save_wm_update_checkpoints", False)):
                snapshot = save_wm_update_snapshot(ckpt_path, log_dir, iteration + 1)
                print(f"[Checkpoint] Saved WM update snapshot: {snapshot}")
            if env_type == "crafter":
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                wm_train_seconds = time.perf_counter() - wm_train_start

            print(
                "[System] World Model updated, reloaded, and synced to Generator."
            )

        # Evaluate held-out post-loss and apply LP + diversity before PPO.
        if is_minigrid:
            gen_interface.finalize_minigrid_rewards()
            # For MiniGrid, expose the first (learning-progress) component of
            # the actual generator reward in the historical CSV slot.  This is
            # the weighted value used by PPO, not the unweighted pre-update
            # focal loss or the aggregate rollout validation loss.
            minigrid_metrics = getattr(gen_interface, "last_minigrid_metrics", {})
            if "Reward_Learning_Progress" in minigrid_metrics:
                gen_val_avg_val_loss_wm = float(
                    minigrid_metrics["Reward_Learning_Progress"]
                )
            gen_loss, gen_entropy, gen_mean_reward = gen_interface.update(iteration=iteration)
            print(
                f"[Generator] Policy Updated. Loss: {gen_loss:.4f} | "
                f"Entropy: {gen_entropy:.4f} | Mean reward: {gen_mean_reward:.4f}"
            )
        else:
            if is_crafter:
                gen_interface.finalize_crafter_learning_progress(apply_rewards=not is_warmup_for_wm)
            elif is_bipedal:
                gen_interface.finalize_bipedal_learning_progress(
                    apply_rewards=not is_warmup_for_wm
                )
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
        target_val_contact_changed_loss = float("nan")
        target_val_focal_loss = 0.0
        target_val_changed_focal_loss = 0.0
        target_val_false_set_rate = 0.0
        target_val_changed_count = 0.0
        target_val_crafter_changed_nll = 0.0
        target_val_crafter_changed_count = 0.0
        target_val_crafter_focal = {name: float("nan") for name in CRAFTER_FOCAL_VAL_METRICS}
        target_val_crafter_inventory = {name: float("nan") for name in CRAFTER_INVENTORY_VAL_METRICS}
        # Validation policy:
        # 1. Skip validation during early warmup to save time.
        # 2. Validate every step afterward to track progress.
        warmup_iters = _safe_int_cfg(
            getattr(cfg.generator_agent, "warmup_iterations", 0),
            default=0,
            name="generator_agent.warmup_iterations",
        )
        if target_files and iteration >= (warmup_iters - 1): 
            if env_type == "crafter" and device.type == "cuda":
                torch.cuda.synchronize(device)
            target_val_start = time.perf_counter()
            print(f"\n>>> Validating on Target Tasks...")
            target_ce_losses = []
            target_inv_losses = []
            target_avg_losses = []
            target_contact_accs = []
            target_contact_bces = []
            target_contact_changed_losses = []
            target_field_loss_values = {
                name: [] for name in MINIGRID_VAL_LOSS_FIELDS
            }
            target_focal_losses = []
            target_changed_focal_losses = []
            target_false_set_rates = []
            target_changed_counts = []
            target_crafter_changed_nlls = []
            target_crafter_changed_counts = []
            target_crafter_focal_values = {name: [] for name in CRAFTER_FOCAL_VAL_METRICS}
            target_crafter_inventory_values = {name: [] for name in CRAFTER_INVENTORY_VAL_METRICS}
            
            # Temporarily switch to validation mode.
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = True

            # Disable W&B during validation to avoid hook errors and run spam.
            old_use_wandb = cfg.attention_model.use_wandb
            cfg.attention_model.use_wandb = False
            validation_sweep = (
                AttentionWM_training.CrafterTargetValidationSweep()
                if env_type == "crafter" else None
            )

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
                    VALID_TIMES=1,
                    validation_sweep=validation_sweep,
                )
                
                if res_dict:
                    if event_audit_enabled:
                        _append_target_event_confusion(
                            target_event_confusion_csv_path, target_event_summary_csv_path,
                            seed, iteration + 1, Path(t_file).stem,
                            res_dict.get("crafter_event_confusion", []),
                        )
                    target_avg_losses.append(res_dict[
                        'original_avg_val_loss_wm' if is_bipedal else 'avg_val_loss_wm'
                    ])
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
                        target_contact_changed_losses.append(res_dict['contact_changed_loss'])
                    elif not is_minigrid:
                        target_ce_losses.append(res_dict.get('terrain_loss', 0.0))
                        target_inv_losses.append(res_dict.get('inventory_loss', 0.0))
                        target_crafter_changed_nlls.append(res_dict.get('changed_nll', 0.0))
                        target_crafter_changed_counts.append(res_dict.get('changed_count', 0.0))
                        for name in CRAFTER_FOCAL_VAL_METRICS:
                            value = res_dict.get(name)
                            if value is not None and np.isfinite(float(value)):
                                target_crafter_focal_values[name].append(float(value))
                        for name in CRAFTER_INVENTORY_VAL_METRICS:
                            value = res_dict.get(name)
                            if value is not None and np.isfinite(float(value)):
                                target_crafter_inventory_values[name].append(float(value))
            
            if validation_sweep is not None:
                validation_sweep.trainer = None
                torch.cuda.empty_cache()
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                target_val_seconds = time.perf_counter() - target_val_start

            # Restore configuration values.
            cfg.attention_model.freeze_weight = old_freeze
            cfg.attention_model.use_wandb = old_use_wandb

            # Aggregate target-task metrics.
            if target_avg_losses:
                target_val_valid_count = len(target_avg_losses)
                target_val_avg_val_loss_wm = float(np.mean(target_avg_losses))
                if is_bipedal:
                    target_val_contact_changed_loss = float(np.mean(target_contact_changed_losses))
                    target_val_contact_acc = float(np.mean(target_contact_accs)) if target_contact_accs else 0.0
                    target_val_contact_bce = float(np.mean(target_contact_bces)) if target_contact_bces else 0.0
                    print(f"[Metrics] Combined Target Loss -> Total: {target_val_avg_val_loss_wm:.4f} | Contact Acc: {target_val_contact_acc:.4f} | Contact BCE: {target_val_contact_bce:.4f}")
                elif not is_minigrid:
                    target_val_val_ce_loss = float(np.mean(target_ce_losses))
                    target_val_val_inv_loss = float(np.mean(target_inv_losses))
                    target_val_crafter_changed_nll = float(np.mean(target_crafter_changed_nlls)) if target_crafter_changed_nlls else 0.0
                    target_val_crafter_changed_count = float(np.mean(target_crafter_changed_counts)) if target_crafter_changed_counts else 0.0
                    target_val_crafter_focal = {name: (float(np.mean(values)) if values else float("nan")) for name, values in target_crafter_focal_values.items()}
                    target_val_crafter_inventory = {
                        name: float(np.mean(values)) if values else float("nan")
                        for name, values in target_crafter_inventory_values.items()
                    }
                    print(
                        f"[Metrics] Combined Target Loss -> Total: "
                        f"{target_val_avg_val_loss_wm:.4f} | "
                        f"Terrain: {target_val_val_ce_loss:.4f} | "
                        f"Changed NLL: {target_val_crafter_changed_nll:.4f} | "
                        f"Changed Count: {target_val_crafter_changed_count:.0f}"
                    )
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
                    target_val_crafter_changed_nll = 0.0
                    target_val_crafter_changed_count = 0.0
        else:
            target_val_avg_val_loss_wm = 0.0
            if is_bipedal:
                target_val_contact_acc = 0.0
                target_val_contact_bce = 0.0
            elif not is_minigrid:
                target_val_val_ce_loss = 0.0
                target_val_val_inv_loss = 0.0
                target_val_crafter_changed_nll = 0.0
                target_val_crafter_changed_count = 0.0

        # --------------------------------------------------------
        # Step 6: Write the experiment summary CSV
        # --------------------------------------------------------
        if summary_csv_path is not None:
            try:
                if (is_minigrid and bool(getattr(diagnostics_cfg, "enabled", False))) or is_crafter:
                    map_records = getattr(
                        gen_interface,
                        "last_minigrid_map_diagnostics" if is_minigrid else "last_crafter_map_diagnostics",
                        [],
                    )
                    if map_records:
                        with open(map_diagnostics_path, mode="a", encoding="utf-8") as handle:
                            for record in map_records:
                                payload = {"Seed": int(seed), "Iter": int(iteration + 1), **record}
                                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                with open(summary_csv_path, mode='a', newline='') as f:
                    # --- [Symmetrical 6-Column Metrics] ---
                    # Ensure we don't log NaN if lists are empty
                    gen_div_reward_val = gen_div_score if gen_div_score is not None else 0.0

                    # Prepare row as a dictionary (Exact Column Order Alignment)
                    ppo_metrics = getattr(gen_interface, "last_generator_update_metrics", {})
                    def ppo_value(name, default=0.0):
                        aliases = {
                            "num_samples": "sample_count",
                            "initial_logprob_error": "initial_logprob_max_error",
                            "grad_norm": "preclip_grad_norm",
                            "param_delta": "parameter_delta",
                        }
                        value = ppo_metrics.get(name, ppo_metrics.get(aliases.get(name, ""), default))
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return default

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
                            "target_val_contact_changed_loss": f"{target_val_contact_changed_loss:.6f}",
                            "target_val_avg_val_loss_wm": f"{target_val_avg_val_loss_wm:.6f}",
                            "Pre_WM_Loss": f"{getattr(gen_interface, 'last_bipedal_metrics', {}).get('Pre_WM_Loss', float('nan')):.6f}",
                            "Post_WM_Loss": f"{getattr(gen_interface, 'last_bipedal_metrics', {}).get('Post_WM_Loss', float('nan')):.6f}",
                            "Learning_Progress": f"{getattr(gen_interface, 'last_bipedal_metrics', {}).get('Learning_Progress', float('nan')):.6f}",
                            "LP_Probe_Count": int(getattr(gen_interface, 'last_bipedal_metrics', {}).get('paired_probe_count', 0)),
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
                                "Learning_Progress": f"{mg_metrics.get('Learning_Progress', 0.0):.6f}",
                                "Difficulty_Rank": f"{mg_metrics.get('Difficulty_Rank', 0.0):.6f}",
                                "Learning_Progress_Rank": f"{mg_metrics.get('Learning_Progress_Rank', 0.0):.6f}",
                                "New_Data_Size": new_data_size,
                                "Buffer_Size": len(fisher_buffer),
                                "Solvable_Count": f"{gen_solvable_count}",
                                "Solvable_Rate": f"{gen_solvable_count / max(1, int(gen_interface.batch_size)):.6f}",
                                "Avg_Path_Len": f"{gen_avg_bfs:.2f}",
                                "Replay_Changed_Fraction": f"{replay_changed_fraction:.6f}",
                                "Batch_Changed_Count": batch_changed_count,
                                "Map_Novelty": f"{mg_metrics.get('Map_Novelty', 0.0):.6f}",
                                "Combination_Novelty": f"{mg_metrics.get('Combination_Novelty', 0.0):.6f}",
                                "Random_Feature_Novelty": f"{mg_metrics.get('Random_Feature_Novelty', 0.0):.6f}",
                                "Pre_Changed_Focal_Loss": f"{mg_metrics.get('Pre_Changed_Focal_Loss', 0.0):.6f}",
                                "Post_Changed_Focal_Loss": f"{mg_metrics.get('Post_Changed_Focal_Loss', 0.0):.6f}",
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
                                "Explorer_Coverage_Rate": f"{mg_metrics.get('Explorer_Coverage_Rate', 0.0):.6f}",
                                "Explorer_Unique_Positions": f"{mg_metrics.get('Explorer_Unique_Positions', 0.0):.2f}",
                                "Explorer_Walkable_Cells": f"{mg_metrics.get('Explorer_Walkable_Cells', 0.0):.2f}",
                                "LP_Split_Spearman": f"{mg_metrics.get('LP_Split_Spearman', 0.0):.6f}",
                                "PPO_Updated": int(bool(ppo_metrics.get("updated", False))),
                                "PPO_Num_Samples": int(ppo_value("num_samples", 0)),
                                "PPO_Policy_Loss": f"{ppo_value('policy_loss'):.6f}",
                                "PPO_Value_Loss": f"{ppo_value('value_loss'):.6f}",
                                "PPO_Approx_KL": f"{ppo_value('approx_kl'):.6f}",
                                "PPO_Clip_Fraction": f"{ppo_value('clip_fraction'):.6f}",
                                "PPO_Ratio_Mean": f"{ppo_value('ratio_mean'):.6f}",
                                "PPO_Ratio_Min": f"{ppo_value('ratio_min'):.6f}",
                                "PPO_Ratio_Max": f"{ppo_value('ratio_max'):.6f}",
                                "PPO_Reward_Std": f"{ppo_value('reward_std'):.6f}",
                                "PPO_Advantage_Std": f"{ppo_value('advantage_std'):.6f}",
                                "PPO_Initial_Logprob_Error": f"{ppo_value('initial_logprob_error'):.9f}",
                                "PPO_Grad_Norm": f"{ppo_value('grad_norm'):.6f}",
                                "PPO_Param_Delta": f"{ppo_value('param_delta'):.9f}",
                            }
                        else:
                            row_data = {
                                "Seed": seed, "Iter": iteration + 1,
                                "New_Data_Size": new_data_size,
                                "Cumulative_Transitions": cumulative_transitions,
                                "Buffer_Size": len(fisher_buffer),
                                "target_val_valid_count": target_val_valid_count,
                                "target_val_sample_count": target_val_valid_count * int(cfg.attention_model.target_validation_max_samples),
                                "target_val_avg_val_loss_wm": f"{target_val_avg_val_loss_wm:.6f}",
                                "target_val_inventory_effect_false_positive_rate": f"{target_val_crafter_inventory['inventory_effect_false_positive_rate']:.6f}",
                                "target_val_inventory_effect_change_recall": f"{target_val_crafter_inventory['inventory_effect_change_recall']:.6f}",
                                "target_val_inventory_effect_change_precision": f"{target_val_crafter_inventory['inventory_effect_change_precision']:.6f}",
                                "target_val_inventory_effect_row_exact": f"{target_val_crafter_inventory['inventory_effect_row_exact']:.6f}",
                                "target_val_changed_focal_loss": f"{target_val_crafter_focal['changed_focal_loss']:.6f}",
                                "target_val_layout_changed_focal_loss": f"{target_val_crafter_focal['layout_changed_focal_loss']:.6f}",
                                "target_val_layout_false_set_rate": f"{target_val_crafter_focal['layout_false_set_rate']:.6f}",
                                "target_val_layout_changed_count": f"{target_val_crafter_focal['layout_changed_count']:.2f}",
                                "target_val_inventory_changed_focal_loss": f"{target_val_crafter_focal['inventory_changed_focal_loss']:.6f}",
                                "target_val_inventory_false_set_rate": f"{target_val_crafter_focal['inventory_false_set_rate']:.6f}",
                                "target_val_inventory_changed_count": f"{target_val_crafter_focal['inventory_changed_count']:.2f}",
                                "target_val_joint_accuracy": f"{target_val_crafter_focal['joint_accuracy']:.6f}",
                                "target_val_position_accuracy": f"{target_val_crafter_focal['position_accuracy']:.6f}",
                                "target_val_direction_accuracy": f"{target_val_crafter_focal['direction_accuracy']:.6f}",
                                "Gen_Mean_Reward": f"{gen_mean_reward:.4f}", "Gen_Loss": f"{gen_loss:.4f}",
                                "Gen_Entropy": f"{gen_entropy:.4f}", "Gen_Div_Reward": f"{gen_div_reward_val:.4f}",
                                "Pre_Changed_Focal_Loss": f"{getattr(gen_interface, 'last_crafter_metrics', {}).get('Pre_Changed_Focal_Loss', float('nan')):.6f}",
                                "Post_Changed_Focal_Loss": f"{getattr(gen_interface, 'last_crafter_metrics', {}).get('Post_Changed_Focal_Loss', float('nan')):.6f}",
                                "Learning_Progress": f"{getattr(gen_interface, 'last_crafter_metrics', {}).get('Learning_Progress', float('nan')):.6f}",
                                **{key: f"{getattr(gen_interface, 'last_crafter_metrics', {}).get(key, float('nan')):.6f}" for key in (
                                    "Pre_Layout_Changed_Focal_Loss", "Post_Layout_Changed_Focal_Loss", "Layout_Learning_Progress",
                                    "Pre_Inventory_Changed_Focal_Loss", "Post_Inventory_Changed_Focal_Loss", "Inventory_Learning_Progress",
                                )},
                                **{key: int(getattr(gen_interface, "last_crafter_metrics", {}).get(key, 0)) for key in (
                                    "Layout_Paired_Probe_Count", "Inventory_Paired_Probe_Count",
                                )},
                                "Solvable_Count": f"{gen_solvable_count}", "Avg_Path_Len": f"{gen_avg_ep_len:.2f}",
                                "Inv_Change_Ratio": f"{inv_change_ratio:.6f}",
                                "Valid_Probe_Count": int(getattr(gen_interface, "last_crafter_metrics", {}).get("valid_probe_count", 0)),
                                "Paired_Probe_Count": int(getattr(gen_interface, "last_crafter_metrics", {}).get("paired_probe_count", 0)),
                                **{key: f"{getattr(gen_interface, 'last_crafter_metrics', {}).get(key, float('nan')):.6f}" for key in (
                                    "Novelty_Total_Mean", "Novelty_Total_Std", "Reward_LP_Mean",
                                    "Reward_Novelty_Mean", "Layout_LP_Scale", "Inventory_LP_Scale",
                                    "Normalized_Layout_LP_Abs_Mean", "Normalized_Inventory_LP_Abs_Mean",
                                    "Layout_Reward_Layout_LP_Abs_Mean", "Layout_Reward_Inventory_LP_Abs_Mean",
                                    "Layout_Reward_Layout_LP_Share", "Layout_Reward_Inventory_LP_Share",
                                    "Layout_Reward_Novelty_Abs_Mean", "Layout_Reward_Novelty_Share",
                                    "Stage_Reward_Layout_LP_Abs_Mean", "Stage_Reward_Inventory_LP_Abs_Mean",
                                    "Stage_Reward_Layout_LP_Share", "Stage_Reward_Inventory_LP_Share",
                                    "Stage_Reward_Novelty_Abs_Mean", "Stage_Reward_Novelty_Share",
                                    "Final_Reward_Mean", "Final_Reward_Std", "Inv_Changed_Slots_Mean",
                                )},
                                "PPO_Reward_Std": f"{ppo_value('reward_std'):.6f}",
                                "PPO_Param_Delta": f"{ppo_value('param_delta'):.9f}",
                                **{key: f"{getattr(gen_interface, 'last_crafter_metrics', {}).get(key, float('nan')):.6f}" for key in (
                                    "Layout_Reward_Mean", "Layout_Reward_Std", "Stage_Reward_Mean",
                                    "Stage_Reward_Std",
                                )},
                                "Stage_No_Inventory_Event_Count": int(getattr(gen_interface, "last_crafter_metrics", {}).get("Stage_No_Inventory_Event_Count", 0)),
                                "PPO_Layout_Policy_Loss": f"{ppo_value('layout_policy_loss'):.6f}",
                                "PPO_Stage_Policy_Loss": f"{ppo_value('stage_policy_loss'):.6f}",
                                "PPO_Layout_Initial_Logprob_Error": f"{ppo_value('initial_layout_logprob_max_error'):.9f}",
                                "PPO_Stage_Initial_Logprob_Error": f"{ppo_value('initial_stage_logprob_max_error'):.9f}",
                                "PPO_Shared_Actor_Gradient_Cosine": f"{ppo_value('shared_actor_gradient_cosine', float('nan')):.6f}",
                                "PPO_Shared_Actor_Gradient_Conflict": f"{ppo_value('shared_actor_gradient_conflict', float('nan')):.0f}",
                                "PPO_Shared_Actor_Gradient_Conflict_Ratio": f"{ppo_value('shared_actor_gradient_conflict_ratio', float('nan')):.6f}",
                                "PPO_Shared_Layout_Gradient_Norm": f"{ppo_value('shared_layout_gradient_norm', float('nan')):.9f}",
                                "PPO_Shared_Stage_Gradient_Norm": f"{ppo_value('shared_stage_gradient_norm', float('nan')):.9f}",
                                "PPO_Stage_Head_Policy_Gradient_Norm": f"{ppo_value('stage_head_policy_gradient_norm', float('nan')):.9f}",
                                "PPO_Stage_Head_Param_Delta": f"{ppo_value('stage_head_parameter_delta', float('nan')):.9f}",
                                "PPO_Stage_Policy_KL_Pre_Post": f"{ppo_value('stage_policy_kl_pre_post', float('nan')):.9f}",
                                "PPO_Stage_Entropy_Pre": f"{ppo_value('stage_entropy_pre', float('nan')):.9f}",
                                "PPO_Stage_Entropy_Post": f"{ppo_value('stage_entropy_post', float('nan')):.9f}",
                                "PPO_Updated": int(bool(ppo_metrics.get("updated", False))),
                                "PPO_Num_Samples": int(ppo_value("num_samples", 0)),
                                "PPO_Policy_Loss": f"{ppo_value('policy_loss'):.6f}",
                                "PPO_Value_Loss": f"{ppo_value('value_loss'):.6f}",
                                "PPO_Approx_KL": f"{ppo_value('approx_kl'):.6f}",
                                "PPO_Clip_Fraction": f"{ppo_value('clip_fraction'):.6f}",
                                "PPO_Ratio_Mean": f"{ppo_value('ratio_mean'):.6f}",
                                "PPO_Ratio_Min": f"{ppo_value('ratio_min'):.6f}",
                                "PPO_Ratio_Max": f"{ppo_value('ratio_max'):.6f}",
                                "PPO_Advantage_Std": f"{ppo_value('advantage_std'):.6f}",
                                "PPO_Initial_Logprob_Error": f"{ppo_value('initial_logprob_error'):.9f}",
                                "PPO_Grad_Norm": f"{ppo_value('grad_norm'):.6f}",
                                **{key: f"{getattr(gen_interface, 'last_crafter_metrics', {}).get(key, 0.0):.6f}" for key in ("Inventory_KEEP_Ratio", *(f"Inventory_Stage_{stage}_{metric}" for stage in range(5) for metric in ("Count", "Mean_LP", "Reward_Std")))},
                            }

                    writer = csv.writer(f)
                    writer.writerow([row_data[column] for column in csv_header] if is_crafter else list(row_data.values()))

                if stage_probe_csv_path is not None:
                    probe_rows = getattr(gen_interface, "last_crafter_map_diagnostics", [])
                    _append_crafter_stage_probe_rows(
                        stage_probe_csv_path, seed, iteration + 1, probe_rows
                    )
                if event_audit_enabled:
                    _append_inventory_event_audit(
                        event_audit_csv_path, seed, iteration + 1, "new_batch",
                        current_inventory_event_rows,
                    )
                    _append_inventory_event_audit(
                        event_audit_csv_path, seed, iteration + 1, "replay_selected",
                        train_res.get("replay_inventory_event_rows", []),
                    )
                    _append_episode_age_event_audit(
                        episode_age_audit_csv_path, seed, iteration + 1,
                        current_episode_age_rollouts,
                    )

            except Exception as e:
                print(f"[Error] Failed to write CSV log: {e}")
                if is_crafter:
                    raise

        if env_type == "crafter":
            timing = gen_interface.crafter_timing
            print(
                "[Timing] "
                f"collection={timing['collection']:.3f}s "
                f"lp_pre={timing['lp_pre']:.3f}s "
                f"wm_train={wm_train_seconds:.3f}s "
                f"lp_post={timing['lp_post']:.3f}s "
                f"target_val={target_val_seconds:.3f}s"
            )

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

        if env_type == "crafter" and bool(getattr(cfg, "save_iteration_state", True)):
            save_state(resume_state_path, {
                "version": 1, "kind": "crafter_mac", "config_digest": config_digest(cfg),
                "completed_iteration": iteration + 1,
                "cumulative_transitions": cumulative_transitions,
                "warmup_cleanup_done": warmup_cleanup_done,
                "wm": wm_instance.state_dict(), "old_params": old_params, "fisher": fisher,
                "replay": replay_state(fisher_buffer),
                "generator": generator_state(gen_interface, include_policy=True),
                "rng": rng_state(),
            })

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
