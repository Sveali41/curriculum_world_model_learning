from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = (
    "dr",
    "mac",
    "mac_full",
    "mac_no_history",
    "mac_no_diversity",
)

METRICS = (
    "target_val_avg_val_loss_wm",
    "target_val_focal_loss",
    "target_val_changed_focal_loss",
    "target_val_false_set_rate",
    "gen_val_avg_val_loss_wm",
    "Combination_Novelty",
    "Random_Feature_Novelty",
    "Pre_Changed_Focal_Loss",
    "Post_Changed_Focal_Loss",
    "Learning_Progress",
    "Difficulty_Rank",
    "Learning_Progress_Rank",
    "Novelty_Rank",
    "Batch_Nearest_Hamming",
    "Archive_Nearest_Hamming",
    "Novelty_Distance_Std",
    "Final_Generator_Reward",
    "Gen_Entropy",
    "Mean_Edit_Rate",
    "Mean_Object_Pair_Distance",
    "Selected_Edit_Pair_Distance",
)


def _result_csv(run_root: Path, variant: str) -> Path:
    files = sorted((run_root / "results" / variant).glob("*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one CSV for {variant}, found: {files}")
    return files[0]


def _load_results(run_root: Path) -> dict[str, pd.DataFrame]:
    results = {}
    for variant in VARIANTS:
        if not (run_root / "results" / variant).is_dir():
            continue
        frame = pd.read_csv(_result_csv(run_root, variant))
        frame = frame.drop_duplicates(["Seed", "Iter"], keep="last").sort_values("Iter")
        results[variant] = frame.reset_index(drop=True)
    if len(results) < 2:
        raise RuntimeError(
            f"Expected results for at least two variants under {run_root / 'results'}"
        )
    return results


def _normalized_auc(frame: pd.DataFrame, metric: str) -> float:
    valid = frame[["Iter", metric]].dropna()
    if len(valid) < 2:
        return float(valid[metric].mean()) if len(valid) else float("nan")
    x = valid["Iter"].to_numpy(dtype=np.float64)
    y = valid[metric].to_numpy(dtype=np.float64)
    return float(np.trapz(y, x) / max(x[-1] - x[0], 1.0))


def _summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, frame in results.items():
        late_count = min(4, len(frame))
        late = frame.tail(late_count)
        row = {
            "Variant": variant,
            "Iterations": len(frame),
            "Late_Window": late_count,
        }
        for metric in METRICS:
            if metric not in frame:
                continue
            row[f"{metric}_late_mean"] = float(late[metric].mean())
            row[f"{metric}_auc"] = _normalized_auc(frame, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(run_root: Path, results: dict[str, pd.DataFrame]) -> None:
    plot_metrics = (
        ("target_val_avg_val_loss_wm", "Natural target loss"),
        ("target_val_focal_loss", "Focal target loss"),
        ("target_val_changed_focal_loss", "Changed focal loss"),
        ("Random_Feature_Novelty", "Random-feature kNN novelty"),
        ("Learning_Progress", "Held-out learning progress"),
        ("Novelty_Rank", "Novelty rank"),
        ("Novelty_Distance_Std", "Novelty distance spread"),
        ("gen_val_avg_val_loss_wm", "Generated-data WM loss"),
        ("Gen_Entropy", "Generator entropy"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharex=True)
    for ax, (metric, title) in zip(axes.flat, plot_metrics):
        for variant, frame in results.items():
            if metric in frame and not frame[metric].isna().all():
                ax.plot(frame["Iter"], frame[metric], marker="o", markersize=2.5, label=variant)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("MiniGrid quick causal diagnostic", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(run_root / "quick_diagnostic_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_environment_montage(run_root: Path, results: dict[str, pd.DataFrame]) -> None:
    max_iter = min(int(frame["Iter"].max()) for frame in results.values()) - 1
    selected_iters = sorted({0, max_iter // 2, max_iter})
    fig, axes = plt.subplots(
        len(results), len(selected_iters),
        figsize=(3.2 * len(selected_iters), 2.8 * len(results)),
        squeeze=False,
    )
    for row, variant in enumerate(results):
        for col, iteration in enumerate(selected_iters):
            ax = axes[row, col]
            image_path = (
                run_root / "environments" / variant
                / f"collect_UED_Dual_iter{iteration}_b0_env.png"
            )
            if image_path.is_file():
                ax.imshow(plt.imread(image_path))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_title(f"{variant} | iter {iteration}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Generated environment comparison (batch 0)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(run_root / "quick_environment_montage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: analyze_minigrid_quick_diagnostics.py RUN_ROOT")
    run_root = Path(sys.argv[1]).expanduser().resolve()
    results = _load_results(run_root)
    summary = _summarize(results)
    summary.to_csv(run_root / "quick_diagnostic_summary.csv", index=False)
    _plot_curves(run_root, results)
    _plot_environment_montage(run_root, results)

    columns = [
        "Variant",
        "target_val_changed_focal_loss_late_mean",
        "target_val_false_set_rate_late_mean",
        "Combination_Novelty_late_mean",
        "Random_Feature_Novelty_late_mean",
        "Batch_Nearest_Hamming_late_mean",
        "Archive_Nearest_Hamming_late_mean",
        "Novelty_Distance_Std_late_mean",
        "gen_val_avg_val_loss_wm_late_mean",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
