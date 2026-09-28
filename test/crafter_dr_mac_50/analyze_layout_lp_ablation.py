"""Compare balanced MAC with and without layout's inventory-LP cross reward."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test/crafter_dr_mac_50/results/mac"
BASELINE_PREFIX = "mac_balanced_ewc20_epoch10_gen1_10plus30_local_seed"
ABLATION_PREFIX = "mac_balanced_ewc20_epoch10_gen1_10plus30_layout_lp_only_local_seed"
METRICS = {
    "target_val_layout_changed_focal_loss": "Layout changed focal loss",
    "target_val_layout_false_set_rate": "Layout false-set rate",
    "target_val_inventory_changed_focal_loss": "Inventory changed focal loss",
    "target_val_inventory_false_set_rate": "Inventory false-set rate",
    "target_val_changed_focal_loss": "Combined changed focal loss",
}


def load_run(root: Path, seed: int) -> pd.DataFrame:
    path = root / "mac_crafter_lp_layout_stage_novelty_results.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "Iter" not in frame or "Seed" not in frame:
        raise ValueError(f"Missing Seed/Iter columns in {path}")
    frame = frame[pd.to_numeric(frame["Seed"], errors="coerce") == seed].copy()
    frame["WM_Update"] = pd.to_numeric(frame["Iter"], errors="coerce") - 10
    return frame[frame["WM_Update"] > 0].sort_values("WM_Update")


def build_summary(runs: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    windows = (("wm10", 10, 10), ("wm20", 20, 20), ("wm26_30", 26, 30))
    for seed in (0, 1):
        for window, first_update, last_update in windows:
            for metric, label in METRICS.items():
                base = runs["baseline", seed]
                ablation = runs["ablation", seed]
                if metric not in base or metric not in ablation:
                    base_mean = ablation_mean = np.nan
                    base_n = ablation_n = 0
                else:
                    base_values = pd.to_numeric(
                        base.loc[base.WM_Update.between(first_update, last_update), metric],
                        errors="coerce",
                    ).dropna()
                    ablation_values = pd.to_numeric(
                        ablation.loc[ablation.WM_Update.between(first_update, last_update), metric],
                        errors="coerce",
                    ).dropna()
                    base_mean = float(base_values.mean()) if len(base_values) else np.nan
                    ablation_mean = float(ablation_values.mean()) if len(ablation_values) else np.nan
                    base_n, ablation_n = len(base_values), len(ablation_values)
                delta = ablation_mean - base_mean
                relative = 100.0 * delta / base_mean if np.isfinite(base_mean) and base_mean != 0 else np.nan
                rows.append({
                    "seed": seed, "window": window, "metric": metric, "label": label,
                    "baseline_mean": base_mean, "ablation_mean": ablation_mean,
                    "delta_ablation_minus_baseline": delta, "relative_delta_percent": relative,
                    "baseline_updates": base_n, "ablation_updates": ablation_n,
                })
    return pd.DataFrame(rows)


def plot_curves(runs: dict[tuple[str, int], pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()
    colors = {"baseline": "#306998", "ablation": "#d17a22"}
    labels = {"baseline": "Cross reward 0.5", "ablation": "Cross reward 0"}
    for ax, (metric, title) in zip(axes, METRICS.items()):
        for seed in (0, 1):
            for arm in ("baseline", "ablation"):
                frame = runs[arm, seed]
                if metric not in frame:
                    continue
                ax.plot(frame.WM_Update, pd.to_numeric(frame[metric], errors="coerce"),
                        color=colors[arm], linestyle="-" if seed == 0 else "--",
                        alpha=0.95 if seed == 0 else 0.65,
                        label=f"{labels[arm]}, seed {seed}")
        ax.axvspan(26, 30, color="#777777", alpha=0.10)
        ax.set_title(title)
        ax.set_xlabel("WM update")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    handles, labels_seen = axes[0].get_legend_handles_labels()
    for ax in axes[:-1]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels_seen:
                handles.append(hh)
                labels_seen.append(ll)
    fig.legend(handles, labels_seen, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("MAC layout LP reward attribution ablation", y=0.995)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_reward_diagnostics(runs: dict[tuple[str, int], pd.DataFrame], output: Path) -> None:
    diagnostics = (
        "Layout_Reward_Inventory_LP_Abs_Mean",
        "Stage_Reward_Layout_LP_Abs_Mean",
        "Stage_Reward_Inventory_LP_Abs_Mean",
    )
    rows = []
    for seed in (0, 1):
        for arm in ("baseline", "ablation"):
            frame = runs[arm, seed]
            recent = frame[frame.WM_Update.between(26, 30)] if "WM_Update" in frame else frame
            row = {"arm": arm, "seed": seed, "wm_update_start": 26, "wm_update_end": 30}
            for name in diagnostics:
                values = pd.to_numeric(recent[name], errors="coerce").dropna() if name in recent else pd.Series(dtype=float)
                row[name] = float(values.mean()) if len(values) else np.nan
                row[f"{name}_n"] = len(values)
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def write_full20_summary(run_roots: dict[tuple[str, int], Path], output: Path) -> None:
    rows = []
    for (arm, seed), run_root in run_roots.items():
        path = run_root / "offline_validation/iter_040_targets_20/aggregate.csv"
        if not path.is_file():
            rows.append({"arm": arm, "seed": seed, "status": "not validated"})
            continue
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle), {})
        row.update({"arm": arm, "seed": seed, "status": "validated"})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "layout_lp_attribution_ablation_analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_roots = {}
    runs = {}
    for seed in (0, 1):
        for arm, prefix in (("baseline", BASELINE_PREFIX), ("ablation", ABLATION_PREFIX)):
            run_id = (
                "mac_balanced_ewc20_epoch10_gen1_10plus30_rerun_local_seed1"
                if arm == "baseline" and seed == 1
                else f"{prefix}{seed}"
            )
            run_root = RESULTS / run_id
            run_roots[arm, seed] = run_root
            runs[arm, seed] = load_run(run_root, seed)
    build_summary(runs).to_csv(args.output_dir / "wm_updates_26_30.csv", index=False)
    plot_curves(runs, args.output_dir / "same_update_curves.png")
    write_reward_diagnostics(runs, args.output_dir / "reward_attribution_wm26_30.csv")
    write_full20_summary(run_roots, args.output_dir / "full20_final.csv")
    available = {
        f"{arm} seed{seed}": int(frame.WM_Update.max()) if len(frame) else 0
        for (arm, seed), frame in runs.items()
    }
    print(f"Saved analysis to {args.output_dir}")
    print(f"Latest WM updates: {available}")


if __name__ == "__main__":
    main()
