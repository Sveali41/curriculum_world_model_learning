"""Short fixed-replay comparison of gamma=0 and gamma=1 Crafter WM losses."""

from __future__ import annotations

import argparse
import copy
import csv
import contextlib
import gc
import json
import os
import random
import sys
from pathlib import Path

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
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from modelBased.data.datamodule import WMRLDataModule
from modelBased.world_model.AttentionWM import AttentionWorldModel
from trainer.common.utils import validate_on_all_targets
from validate_checkpoint import _check_saved_config

SUBSET_SIZE = 4096
SUBSET_SEED = 3026
TRAIN_SEED = 9137
EXPECTED_UPDATE = {"dr": 33, "mac": 60}
METRICS = (
    "avg_val_loss_wm",
    "changed_nll",
    "changed_focal_loss",
    "layout_changed_focal_loss",
    "layout_false_set_rate",
    "layout_changed_count",
    "inventory_changed_focal_loss",
    "inventory_false_set_rate",
    "inventory_changed_count",
    "inventory_effect_change_recall",
    "inventory_effect_change_precision",
    "inventory_effect_false_positive_rate",
    "inventory_effect_row_exact",
)


def _compose_config(arm: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(
            config_name=f"config_{arm}_crafter_dr_mac_50",
            overrides=["seed=0"],
        )


def _run_dir(cfg, arm: str) -> Path:
    name = "dr_log_dir" if arm == "dr" else "mac_results_dir"
    return Path(str(getattr(cfg, name))).expanduser().resolve()


def _target_spec(cfg):
    domain = cfg.domains.crafter
    directory = Path(str(getattr(domain, "val_data_path", domain.target_tasks_folder))).resolve()
    prefix = str(getattr(domain, "val_task_prefix", domain.target_task_prefix))
    suffix = str(getattr(domain, "val_suffix", domain.target_task_suffix))
    start = int(getattr(domain, "val_start_idx", domain.target_task_start_idx))
    count = int(domain.target_task_count)
    names = [f"{prefix}{idx}" for idx in range(start, start + count)]
    missing = [directory / f"{name}{suffix}" for name in names if not (directory / f"{name}{suffix}").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} held-out target archives; first: {missing[0]}")
    return directory, names, suffix


def _replay_subset(state, output_dir: Path):
    replay = state["replay"]["buffer"]
    if len(replay) < SUBSET_SIZE:
        raise ValueError(f"Replay buffer has {len(replay)} rows; need {SUBSET_SIZE}")
    indices = np.random.default_rng(SUBSET_SEED).choice(
        len(replay), size=SUBSET_SIZE, replace=False
    )
    fields = {
        "a": "obs",
        "b": "obs_next",
        "c": "act",
        "e": "done",
        "g": "inv",
        "h": "inv_next",
    }
    data = {
        key: np.stack([np.asarray(replay[int(idx)][source]) for idx in indices])
        for key, source in fields.items()
    }
    changed = {
        "layout_object_rows": int(np.any(data["a"][:, 0] != data["b"][:, 0], axis=(1, 2)).sum()),
        "layout_direction_rows": int(np.any(data["a"][:, 1] != data["b"][:, 1], axis=(1, 2)).sum()),
        "inventory_change_rows": int(np.any(data["g"][:, 4:] != data["h"][:, 4:], axis=1).sum()),
    }
    archive_path = output_dir / "replay_subset_4096.npz"
    np.savez_compressed(archive_path, **data)
    with (output_dir / "replay_indices.json").open("w") as handle:
        json.dump({"subset_seed": SUBSET_SEED, "indices": indices.tolist()}, handle)
    return data, indices, changed, archive_path


def _set_gamma(cfg, gamma: float):
    OmegaConf.update(cfg, "domains.crafter.focal_gamma", float(gamma), force_add=True)
    OmegaConf.resolve(cfg)


def _evaluate(cfg, state_dict, target_dir, target_names, suffix, label: str, log_path: Path):
    eval_cfg = copy.deepcopy(cfg)
    _set_gamma(eval_cfg, 0.0)
    eval_cfg.attention_model.use_wandb = False
    OmegaConf.update(eval_cfg, "attention_model.enable_progress_bar", False, force_add=True)
    net = AttentionWorldModel(eval_cfg.attention_model)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    with log_path.open("w") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            result = validate_on_all_targets(
                eval_cfg,
                net,
                str(target_dir),
                target_names,
                suffix,
                phase_name=f"focal_ablation_{label}",
                VALID_TIMES=1,
                disable_wandb=True,
            )
    if int(result.get("valid_count", 0)) != 20:
        raise RuntimeError(f"Expected validation on 20 targets, got {result.get('valid_count')}")
    summary = {name: float(result.get(name, float("nan"))) for name in METRICS}
    summary["per_target"] = {
        task: {name: float(metrics.get(name, float("nan"))) for name in METRICS}
        for task, metrics in result["per_target"].items()
    }
    del net, eval_cfg
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _train_one(cfg, state, data, gamma: float, output_dir: Path, arm: str):
    train_cfg = copy.deepcopy(cfg)
    _set_gamma(train_cfg, gamma)
    train_cfg.attention_model.n_epochs = 1
    train_cfg.attention_model.batch_size = 64
    train_cfg.attention_model.n_cpu = 0
    train_cfg.attention_model.use_wandb = False
    train_cfg.attention_model.visualization = False
    OmegaConf.update(train_cfg, "attention_model.enable_progress_bar", False, force_add=True)

    net = AttentionWorldModel(train_cfg.attention_model)
    net.load_state_dict(state["wm"], strict=True)
    net.set_consolidation(state["old_params"], state["fisher"], load_weights=False)
    datamodule = WMRLDataModule(hparams=train_cfg.attention_model, data=data)

    pl.seed_everything(TRAIN_SEED, workers=True)
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=32,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        deterministic=False,
        log_every_n_steps=20,
    )
    with (output_dir / "training.log").open("w") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            trainer.fit(net, datamodule=datamodule)
    state_dict = {key: value.detach().cpu() for key, value in net.state_dict().items()}
    checkpoint_path = output_dir / f"{arm}_seed0_gamma{int(gamma)}.pt"
    torch.save(state_dict, checkpoint_path)
    result = {
        "gamma": gamma,
        "epochs": 1,
        "batch_size": 64,
        "train_samples": int(len(datamodule.data_train)),
        "validation_samples": int(len(datamodule.data_test)),
        "checkpoint": str(checkpoint_path),
    }
    del trainer, net, datamodule, train_cfg
    gc.collect()
    torch.cuda.empty_cache()
    return state_dict, result


def _write_json(path: Path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)


def _write_comparison_csv(path: Path, results):
    columns = ("arm", "phase", "starting_update", *METRICS[:-1])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for arm, values in results.items():
            for phase in ("initial", "gamma0", "gamma1"):
                metrics = values[phase]
                writer.writerow({
                    "arm": arm,
                    "phase": phase,
                    "starting_update": values["starting_update"],
                    **{name: metrics[name] for name in METRICS[:-1]},
                })
            writer.writerow({
                "arm": arm,
                "phase": "gamma1_minus_gamma0",
                "starting_update": values["starting_update"],
                **{name: values["gamma1_minus_gamma0"][name] for name in METRICS[:-1]},
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=("dr", "mac"), default=("dr", "mac"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test/crafter_dr_mac_50/results/focal_ablation_seed0_full20",
    )
    args = parser.parse_args()
    output_root = args.output.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty focal ablation directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for arm in args.arms:
        cfg = _compose_config(arm)
        run_dir = _run_dir(cfg, arm)
        _check_saved_config(cfg, run_dir)
        state_path = run_dir / f"crafter_{arm}_seed0.resume.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing saved {arm.upper()} resume state: {state_path}")
        state = torch.load(str(state_path), map_location="cpu", weights_only=False)
        expected = EXPECTED_UPDATE[arm]
        if int(state["completed_iteration"]) != expected:
            raise ValueError(
                f"Expected {arm.upper()} resume state at update {expected}, "
                f"got {state['completed_iteration']}"
            )
        target_dir, target_names, suffix = _target_spec(cfg)
        arm_dir = output_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        data, indices, changed, archive_path = _replay_subset(state, arm_dir)
        cfg.attention_model.target_validation_max_samples = 500
        cfg.attention_model.target_validation_seed = int(getattr(cfg, "seed", 0))

        log_dir = arm_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        initial_metrics = _evaluate(
            cfg, state["wm"], target_dir, target_names, suffix, f"{arm}_initial",
            log_dir / "initial_validation.log",
        )
        arm_results = {
            "arm": arm,
            "seed": 0,
            "resume_state": str(state_path),
            "starting_update": expected,
            "replay_subset": str(archive_path),
            "replay_subset_size": int(len(indices)),
            "replay_subset_seed": SUBSET_SEED,
            "replay_subset_change_counts": changed,
            "validation_target_count": 20,
            "validation_samples_per_target": 500,
            "validation_seed": int(getattr(cfg, "seed", 0)),
            "initial_metrics_gamma0_eval": initial_metrics,
            "trained": {},
        }
        for gamma in (0.0, 1.0):
            output_dir = arm_dir / f"gamma_{int(gamma)}"
            output_dir.mkdir()
            _write_json(output_dir / "replay_indices.json", {
                "subset_seed": SUBSET_SEED,
                "indices": indices.tolist(),
            })
            state_dict, train_info = _train_one(
                cfg, state, data, gamma, output_dir, arm
            )
            post_metrics = _evaluate(
                cfg, state_dict, target_dir, target_names, suffix,
                f"{arm}_gamma{int(gamma)}_post",
                log_dir / f"gamma_{int(gamma)}_validation.log",
            )
            arm_results["trained"][str(int(gamma))] = {
                **train_info,
                "metrics_gamma0_eval": post_metrics,
            }
            _write_json(output_dir / "metrics.json", {
                "arm": arm,
                "starting_update": expected,
                "gamma": gamma,
                "initial_metrics_gamma0_eval": initial_metrics,
                "post_metrics_gamma0_eval": post_metrics,
                "training": train_info,
                "replay_subset_change_counts": changed,
            })
            del state_dict

        gamma0 = arm_results["trained"]["0"]["metrics_gamma0_eval"]
        gamma1 = arm_results["trained"]["1"]["metrics_gamma0_eval"]
        arm_results["gamma1_minus_gamma0"] = {
            name: float(gamma1[name] - gamma0[name])
            for name in METRICS if name != "per_target"
        }
        _write_json(arm_dir / "summary.json", arm_results)
        all_results[arm] = {
            "starting_update": expected,
            "change_counts": changed,
            "initial": initial_metrics,
            "gamma0": gamma0,
            "gamma1": gamma1,
            "gamma1_minus_gamma0": arm_results["gamma1_minus_gamma0"],
        }
        del state, data, cfg
        gc.collect()
        torch.cuda.empty_cache()

    _write_json(output_root / "summary.json", all_results)
    _write_comparison_csv(output_root / "comparison.csv", all_results)
    print(json.dumps({
        arm: {
            "starting_update": values["starting_update"],
            "gamma0": values["gamma0"],
            "gamma1": values["gamma1"],
            "gamma1_minus_gamma0": values["gamma1_minus_gamma0"],
        }
        for arm, values in all_results.items()
    }, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
