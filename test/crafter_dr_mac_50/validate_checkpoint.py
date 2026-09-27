"""Validate a retained Crafter DR/MAC WM update on fixed uniform targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WM_ROOT = ROOT / "wm"
CONFIG_DIR = Path(__file__).resolve().parent / "conf"
os.environ.setdefault("TRAINER_ROOT", str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
os.environ.setdefault("WM_ROOT", str(WM_ROOT))
os.environ.setdefault("ENV_PATH", str(ROOT / "level"))
os.environ.setdefault("WORLD_MODEL_PATH", str(WM_ROOT / "modelBased"))
os.environ.setdefault("TRAINER_PATH", str(ROOT / "trainer"))
sys.path.insert(0, str(WM_ROOT))
sys.path.insert(1, str(ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from modelBased.world_model.AttentionWM import AttentionWorldModel
from trainer.common.utils import set_seed, validate_on_all_targets


def _experiment_kind(config_name: str) -> str:
    if config_name.startswith("config_dr_"):
        return "dr"
    if config_name.startswith("config_mac_"):
        return "mac"
    raise ValueError("Use a Crafter DR or MAC experiment config from this experiment folder")


def _check_saved_config(cfg, run_dir: Path) -> None:
    saved_path = run_dir / "hydra" / ".hydra" / "config.yaml"
    if not saved_path.is_file():
        raise FileNotFoundError(f"Missing original experiment config: {saved_path}")
    original = OmegaConf.load(saved_path)
    if int(original.seed) != int(cfg.seed):
        raise ValueError(f"Requested seed {cfg.seed} differs from saved run seed {original.seed}")
    for name in ("attention_model",):
        saved = OmegaConf.to_container(original[name], resolve=True)
        requested = OmegaConf.to_container(cfg[name], resolve=True)
        for values in (saved, requested):
            values.pop("model_save_path", None)
            values.pop("data_dir", None)
        if saved != requested:
            raise ValueError(f"{name} differs from the saved run config: {saved_path}")
    saved_wm = OmegaConf.to_container(original.domains.crafter.crafter_wm, resolve=True)
    requested_wm = OmegaConf.to_container(cfg.domains.crafter.crafter_wm, resolve=True)
    if saved_wm != requested_wm:
        raise ValueError(f"Crafter WM config differs from the saved run config: {saved_path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="Experiment Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Experiment YAML name without .yaml")
    parser.add_argument("--iteration", type=int, required=True, help="One-based DR/MAC iteration with a saved WM update")
    parser.add_argument("--target-count", type=int, default=20, help="Targets to validate; default 20 (use 1 for smoke)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as seed=1 or dr_quick_run_id=...")
    args = parser.parse_args()
    if args.iteration < 1 or not 1 <= args.target_count <= 20:
        parser.error("--iteration must be positive and --target-count must be 1–20")
    kind = _experiment_kind(args.config_name)
    config_dir = args.config_dir.resolve()
    if not (config_dir / f"{args.config_name}.yaml").is_file():
        parser.error(f"Missing experiment config: {config_dir / (args.config_name + '.yaml')}")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    if str(cfg.domain) != "crafter":
        raise ValueError(f"Expected Crafter config, got domain={cfg.domain}")
    run_dir = Path(str(cfg.dr_log_dir if kind == "dr" else cfg.mac_results_dir)).resolve()
    _check_saved_config(cfg, run_dir)
    checkpoint = run_dir / "wm_snapshots" / f"iter_{args.iteration:03d}.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WM update snapshot is missing: {checkpoint}")
    domain = cfg.domains.crafter
    target_dir = Path(str(getattr(domain, "val_data_path", domain.target_tasks_folder))).resolve()
    prefix = str(getattr(domain, "val_task_prefix", domain.target_task_prefix))
    suffix = str(getattr(domain, "val_suffix", domain.target_task_suffix))
    start = int(getattr(domain, "val_start_idx", domain.target_task_start_idx))
    names = [f"{prefix}{index}" for index in range(start, start + args.target_count)]
    missing = [target_dir / f"{name}{suffix}" for name in names if not (target_dir / f"{name}{suffix}").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} fixed uniform target(s), first: {missing[0]}")
    if int(cfg.attention_model.target_validation_max_samples) != 500:
        raise ValueError("This comparison requires attention_model.target_validation_max_samples=500")

    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    model = AttentionWorldModel(cfg.attention_model)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    summary = validate_on_all_targets(
        cfg, model, str(target_dir), names, suffix,
        phase_name=f"offline_{kind}_iter_{args.iteration:03d}", VALID_TIMES=1,
    )
    if int(summary["valid_count"]) != args.target_count or set(summary["per_target"]) != set(names):
        raise RuntimeError(f"Invalid target validation count: {summary['valid_count']}/{args.target_count}")
    if not all(np.isfinite(float(item["avg_val_loss_wm"])) for item in summary["per_target"].values()):
        raise RuntimeError("Non-finite target validation loss")

    output_dir = run_dir / "offline_validation" / f"iter_{args.iteration:03d}_targets_{args.target_count:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "seed": int(cfg.seed), "iteration": args.iteration, "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "target_count": args.target_count,
        "samples_per_target": 500, "target_subset_seed": int(cfg.attention_model.target_validation_seed),
    }
    scalar = lambda values: {key: value for key, value in values.items() if isinstance(value, (int, float, str))}
    _write_csv(output_dir / "aggregate.csv", [{**identity, **scalar(summary)}])
    _write_csv(
        output_dir / "per_target.csv",
        [{**identity, "target": name, **scalar(summary["per_target"][name])} for name in names],
    )
    print(f"[Crafter WM validation] {summary['valid_count']} uniform targets × 500 samples")
    print(f"[Crafter WM validation] checkpoint sha256={checkpoint_hash}")
    print(f"[Crafter WM validation] results={output_dir}")


if __name__ == "__main__":
    main()
