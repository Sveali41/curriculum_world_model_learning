"""Evaluate retained Crafter WM updates on all 20 uniform targets into one CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import time

from validate_checkpoint import (
    CONFIG_DIR,
    ROOT,
    _check_saved_config,
    hydra,
    np,
    OmegaConf,
    set_seed,
    torch,
    AttentionWorldModel,
    validate_on_all_targets,
)


FIELDS = (
    "Seed",
    "WM_Update",
    "Iter",
    "checkpoint_sha256",
    "target_count",
    "samples_per_target",
    "target_subset_seed",
    "target_val_layout_changed_focal_loss",
    "target_val_inventory_changed_focal_loss",
    "target_val_changed_focal_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dr", "mac", "target", "p2e"), default="mac")
    parser.add_argument("--seeds", nargs="+", type=int, choices=tuple(range(5)), default=(0, 1))
    parser.add_argument("--start-update", type=int, default=1)
    parser.add_argument("--end-update", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    max_update = {"dr": 50, "mac": 50, "target": 20, "p2e": 100}[args.arm]
    end_update = args.end_update if args.end_update is not None else max_update
    if not 1 <= args.start_update <= end_update <= max_update:
        parser.error(f"Expected 1 <= --start-update <= --end-update <= {max_update}")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    config_name = f"config_{args.arm}_crafter_dr_mac_50"
    iteration_offset = 10 if args.arm == "mac" else 0

    configs = {}
    checkpoints = {}
    for seed in args.seeds:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = hydra.compose(
                config_name=config_name, overrides=[f"seed={seed}"]
            )
        run_root = {
            "dr": "dr_log_dir", "mac": "mac_results_dir",
            "target": "target_dr_mac_run_root", "p2e": "p2e_dr_mac_run_root",
        }[args.arm]
        run_dir = Path(str(getattr(cfg, run_root))).resolve()
        saved_config = run_dir / "hydra" / ".hydra" / "config.yaml"
        if saved_config.is_file():
            _check_saved_config(cfg, run_dir)
        elif args.arm != "p2e":
            raise FileNotFoundError(f"Missing original experiment config: {saved_config}")
        else:
            print(f"[Full20] P2E seed={seed}: saved config missing; checking checkpoint metadata", flush=True)
        if int(cfg.attention_model.target_validation_max_samples) != 500:
            raise ValueError("Full validation requires 500 fixed samples per target")
        domain = cfg.domains.crafter
        target_dir = Path(str(getattr(domain, "val_data_path", domain.target_tasks_folder))).resolve()
        prefix = str(getattr(domain, "val_task_prefix", domain.target_task_prefix))
        suffix = str(getattr(domain, "val_suffix", domain.target_task_suffix))
        start = int(getattr(domain, "val_start_idx", domain.target_task_start_idx))
        names = [f"{prefix}{index}" for index in range(start, start + 20)]
        missing = [target_dir / f"{name}{suffix}" for name in names if not (target_dir / f"{name}{suffix}").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} uniform targets; first: {missing[0]}")
        configs[seed] = (cfg, target_dir, names, suffix, saved_config.is_file())
        for update in range(args.start_update, end_update + 1):
            iteration = update + iteration_offset
            checkpoint = run_dir / "wm_snapshots" / f"iter_{iteration:03d}.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing {args.arm.upper()} WM update {update}: {checkpoint}")
            checkpoints[seed, update] = checkpoint

    output = (args.output or ROOT / f"test/crafter_dr_mac_50/results/{args.arm}/full20_changed_focal.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output.is_file() or output.stat().st_size == 0
    completed = {}
    if not needs_header:
        with output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Existing CSV has a different header: {output}")
            for row in reader:
                key = (int(row["Seed"]), int(row["WM_Update"]))
                if key in completed:
                    raise ValueError(f"Duplicate completed WM update {key} in {output}")
                completed[key] = row

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
            handle.flush()
        for seed in args.seeds:
            cfg, target_dir, names, suffix, has_saved_config = configs[seed]
            for update in range(args.start_update, end_update + 1):
                checkpoint = checkpoints[seed, update]
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                key = (seed, update)
                if key in completed:
                    previous = completed[key]
                    if (
                        previous["checkpoint_sha256"] != checkpoint_hash
                        or int(previous["Iter"]) != update + iteration_offset
                        or int(previous["target_count"]) != 20
                        or int(previous["samples_per_target"]) != 500
                        or int(previous["target_subset_seed"])
                        != int(cfg.attention_model.target_validation_seed)
                    ):
                        raise ValueError(f"Existing row {key} does not match checkpoint or validation protocol")
                    print(f"[Full20] Skipping completed seed={seed} update={update}", flush=True)
                    continue

                started = time.perf_counter()
                set_seed(seed)
                raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
                if not has_saved_config:
                    expected = OmegaConf.to_container(cfg.attention_model, resolve=True)
                    embedded = OmegaConf.to_container(raw["hyper_parameters"], resolve=True)
                    for values in (expected, embedded):
                        values.pop("model_save_path", None)
                        values.pop("data_dir", None)
                    if embedded != expected:
                        raise ValueError(f"Checkpoint model config differs from validation config: {checkpoint}")
                model = AttentionWorldModel(cfg.attention_model)
                if not has_saved_config and raw.get("world_model_contract") != model.model.checkpoint_contract:
                    raise ValueError(f"Checkpoint world-model contract differs from validation config: {checkpoint}")
                model.load_state_dict(state, strict=True)
                model = model.to(device).eval()
                summary = validate_on_all_targets(
                    cfg,
                    model,
                    str(target_dir),
                    names,
                    suffix,
                    phase_name=f"offline_{args.arm}_seed{seed}_update{update:02d}_full20",
                    VALID_TIMES=1,
                )
                del model
                if int(summary["valid_count"]) != 20 or set(summary["per_target"]) != set(names):
                    raise RuntimeError(f"Incomplete full20 validation for seed={seed} update={update}")
                metrics = {
                    f"target_val_{name}": float(summary[name])
                    for name in (
                        "layout_changed_focal_loss",
                        "inventory_changed_focal_loss",
                        "changed_focal_loss",
                    )
                }
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError(f"Nonfinite full20 focal loss for seed={seed} update={update}")
                writer.writerow(
                    {
                        "Seed": seed,
                        "WM_Update": update,
                        "Iter": update + iteration_offset,
                        "checkpoint_sha256": checkpoint_hash,
                        "target_count": 20,
                        "samples_per_target": 500,
                        "target_subset_seed": int(cfg.attention_model.target_validation_seed),
                        **metrics,
                    }
                )
                handle.flush()
                print(
                    f"[Full20] arm={args.arm} seed={seed} update={update}/{max_update} "
                    f"combined={metrics['target_val_changed_focal_loss']:.6f} "
                    f"layout={metrics['target_val_layout_changed_focal_loss']:.6f} "
                    f"inventory={metrics['target_val_inventory_changed_focal_loss']:.6f} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    print(f"[Full20] CSV: {output}")


if __name__ == "__main__":
    main()
