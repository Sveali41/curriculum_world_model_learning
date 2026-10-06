"""Compare five-seed MiniGrid Random-policy MAC, DR, and PUS runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRIC = "target_val_changed_focal_loss"
MAC_CSV = ROOT / "outputs/results/mac/minigrid_ued_results_mask5_focal_reservoir_sa.csv"
DR_CSV = ROOT / "outputs/results/dr/dr_summary_minigrid_mask5_focal_reservoir.csv"
PUS_CSV = "pus_summary_minigrid_mask5_focal_reservoir.csv"


def _read_seed(csv_path, seed, warmup):
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    rows = frame.loc[frame["Seed"] == seed].sort_values("Iter").copy()
    if rows["Iter"].duplicated().any():
        raise ValueError(f"Duplicate iterations for seed {seed}: {csv_path}")
    if not np.array_equal(rows["Iter"].to_numpy(), np.arange(1, warmup + 51)):
        raise ValueError(f"Expected {warmup + 50} collection rounds for seed {seed}: {csv_path}")
    if not np.all(rows["New_Data_Size"].to_numpy(dtype=int) == 2800):
        raise ValueError(f"Expected 2800 collected transitions per round: {csv_path}")
    rows = rows.loc[rows["Iter"] > warmup]
    expected_iterations = np.arange(warmup + 1, warmup + 51)
    if not np.array_equal(rows["Iter"].to_numpy(), expected_iterations):
        raise ValueError(f"Expected 50 completed WM updates for seed {seed}: {csv_path}")
    cumulative_wm_transitions = rows["New_Data_Size"].cumsum().to_numpy(dtype=int)
    values = rows[METRIC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing {METRIC} for seed {seed}: {csv_path}")
    return values, cumulative_wm_transitions


def _check_pus_selection(seed_dir):
    path = seed_dir / "results/pus_selected_settings.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    if len(rows) != 400 or rows.groupby("Iter").size().to_dict() != dict.fromkeys(range(1, 51), 8):
        raise ValueError(f"Expected 8 selected maps in each of 50 iterations: {path}")
    if (rows.loc[rows["Iter"] == 1, "Selection_Mode"] != "uniform").any():
        raise ValueError(f"PUS first iteration must sample uniformly: {path}")
    if not rows["Selection_Mode"].isin(("uniform", "uncertainty")).all():
        raise ValueError(f"Unexpected PUS selection mode: {path}")
    for name in ("door", "key"):
        color_columns = [f"{name}_{color}" for color in ("yellow", "red", "blue", "green")]
        if not (rows[color_columns].sum(axis=1) == rows[f"n_{name}"]).all():
            raise ValueError(f"PUS {name} color counts do not match map parameters: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pus_root", type=Path, help="Directory containing seed0/ ... seed4/")
    parser.add_argument("--mac-csv", type=Path, default=MAC_CSV)
    parser.add_argument("--dr-csv", type=Path, default=DR_CSV)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    methods = {
        "DR": (args.dr_csv, 0),
        "PUS": (None, 0),
        "MAC": (args.mac_csv, 10),
    }
    curves = []
    finals = []
    plotted = {}
    pus_selections = []
    for method, (shared_csv, warmup) in methods.items():
        seed_values = []
        wm_transitions_by_seed = []
        for seed in range(5):
            seed_dir = args.pus_root / f"seed{seed}"
            csv_path = seed_dir / "results" / PUS_CSV if method == "PUS" else shared_csv
            values, wm_transitions = _read_seed(csv_path, seed, warmup)
            if method == "PUS":
                pus_selections.append(_check_pus_selection(seed_dir))
            seed_values.append(values)
            wm_transitions_by_seed.append(wm_transitions)
        values = np.stack(seed_values)
        wm_transitions = np.stack(wm_transitions_by_seed)
        if not np.all(wm_transitions == wm_transitions[0]):
            raise ValueError(f"WM training data budgets differ between {method} seeds")
        means = values.mean(axis=0)
        sds = values.std(axis=0, ddof=1)
        plotted[method] = (means, sds, wm_transitions[0])
        for update, (mean, sd, total) in enumerate(zip(means, sds, wm_transitions[0]), start=1):
            curves.append({
                "Method": method, "WM_Update": update,
                "Cumulative_WM_Transitions": total,
                "Changed_Focal_Loss_Mean": mean,
                "Changed_Focal_Loss_SD": sd,
            })
        finals.append({
            "Method": method, "Seeds": 5,
            "WM_Updates": 50,
            "Cumulative_WM_Transitions": int(wm_transitions[0, -1]),
            "Final_Changed_Focal_Loss_Mean": means[-1],
            "Final_Changed_Focal_Loss_SD": sds[-1],
        })

    output_dir = args.output_dir or args.pus_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(output_dir / "changed_focal_curves.csv", index=False)
    final_frame = pd.DataFrame(finals)
    final_frame.to_csv(output_dir / "changed_focal_final.csv", index=False)
    selections = pd.concat(pus_selections, ignore_index=True)
    color_columns = [
        f"{name}_{color}"
        for name in ("door", "key")
        for color in ("yellow", "red", "blue", "green")
    ]
    color_coverage = selections.groupby("Seed")[color_columns].sum()
    color_coverage["Maps_With_Multiple_Colors"] = selections.groupby("Seed")[
        "Unique_Key_Door_Colors"
    ].apply(lambda values: int((values >= 2).sum()))
    color_coverage.to_csv(output_dir / "pus_color_coverage.csv")

    colors = {"DR": "#D55E00", "PUS": "#009E73", "MAC": "#0072B2"}
    fig, ax = plt.subplots(figsize=(6.0, 3.5), constrained_layout=True)
    for method in methods:
        mean, sd, wm_transitions = plotted[method]
        ax.plot(wm_transitions, mean, label=method, color=colors[method], linewidth=1.8)
        ax.fill_between(wm_transitions, mean - sd, mean + sd, color=colors[method], alpha=0.14)
    ax.set_xlabel("Cumulative WM training transitions")
    ax.set_ylabel("Target changed focal loss ↓")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "changed_focal_comparison.pdf")
    fig.savefig(output_dir / "changed_focal_comparison.png", dpi=300)
    plt.close(fig)
    print(final_frame.to_string(index=False))
    print(f"Saved comparison to {output_dir}")


if __name__ == "__main__":
    main()
