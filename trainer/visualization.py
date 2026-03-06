import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================================
# 保持你原来的格式
# ==========================================================

sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (12, 10),
    "lines.linewidth": 2.5,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ==========================================================
# 自动识别 seed 列
# ==========================================================

def get_seed_column(df):
    if "Seed" in df.columns:
        return "Seed"
    if "seed" in df.columns:
        return "seed"
    df["Seed"] = 0
    return "Seed"

# ==========================================================
# 插值 + seed mean/std
# ==========================================================

def compute_seed_mean_std_interpolated(df, loss_col, seed_col, grid_points=200):

    seeds = df[seed_col].unique()

    mins, maxs = [], []
    for s in seeds:
        d = df[df[seed_col] == s]
        mins.append(d["Transitions"].min())
        maxs.append(d["Transitions"].max())

    global_min = max(mins)
    global_max = min(maxs)

    grid = np.linspace(global_min, global_max, grid_points)

    all_curves = []

    for s in seeds:
        d = df[df[seed_col] == s]
        x = d["Transitions"].values
        y = d[loss_col].values
        interp_y = np.interp(grid, x, y)
        all_curves.append(interp_y)

    all_curves = np.array(all_curves)

    mean = np.mean(all_curves, axis=0)
    std = np.std(all_curves, axis=0)

    return grid, mean, std

# ==========================================================
# 主函数
# ==========================================================

def plot_three_methods_warmup_aligned(
    ued_path,
    dr_path,
    baseline_path,
    warmup_iter=10,
    max_k=150  # 横轴最大值（k transitions）
):

    max_transition = max_k * 1000

    methods = {
        "MAC": {
            "path": ued_path,
            "color": "#1f77b4",
            "style": "-"
        },
        "DR Baseline": {
            "path": dr_path,
            "color": "#d62728",
            "style": "--"
        },
        "Baseline": {
            "path": baseline_path,
            "color": "#2ca02c",
            "style": "-."
        }
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, config in methods.items():

        p = config["path"]
        if not os.path.exists(p) and os.path.exists("trainer/" + p):
            p = "trainer/" + p

        if not os.path.exists(p):
            print(f"[Warning] Missing: {p}")
            continue

        df = pd.read_csv(p)
        seed_col = get_seed_column(df)

        # =========================================
        # Baseline
        # =========================================
        if "transitions" in df.columns:

            loss_col = "avg_target_loss"
            df["Transitions"] = df["transitions"]

        # =========================================
        # UED / DR
        # =========================================
        else:

            loss_col = "WM_Val_Loss"
            df = df.sort_values([seed_col, "Iter"])

            if label == "MAC":
                # 只对 MAC 过滤 warmup
                df = df[df["Iter"] > warmup_iter].copy()

            df["Transitions"] = df.groupby(seed_col)["New_Data_Size"].cumsum()

        # =========================================
        # 限制最大 transition（统计也裁剪）
        # =========================================
        df = df[df["Transitions"] <= max_transition].copy()

        if len(df) == 0:
            continue

        # =========================================
        # 计算 mean/std
        # =========================================

        grid, mean, std = compute_seed_mean_std_interpolated(
            df,
            loss_col,
            seed_col
        )

        x = grid / 1000  # k transitions

        ax.plot(
            x,
            mean,
            label=label,
            color=config["color"],
            linestyle=config["style"],
            linewidth=2.5
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=config["color"],
            alpha=0.15
        )

    ax.set_title("Generalization Performance Comparison", fontweight='bold')
    ax.set_xlabel(r"Transitions ($\times 10^3$)")
    ax.set_ylabel("Target Task Loss (Log Scale)")
    ax.set_yscale("log")
    ax.set_xlim(0, max_k)  # 限制横轴显示
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()

    save_path = "three_method_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")

# ==========================================================
# main
# ==========================================================

if __name__ == "__main__":

    f_ued = "logs/results/experiment_summary_mask3_mse_weighted.csv"
    f_dr = "logs/results_dr/experiment_summary_dr_mask_3_mse_weighted.csv"
    f_baseline = "logs/results/target_loss_baseline_ued.csv"

    plot_three_methods_warmup_aligned(
        f_ued,
        f_dr,
        f_baseline,
        warmup_iter=10,
        max_k=150
    )