import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Paths
# -----------------------------
csv_mac_path = os.path.join(SCRIPT_DIR, 'logs', 'results', 'crafter_ued_results_mask5.csv')
csv_no_his_path = os.path.join(SCRIPT_DIR, 'logs', 'results', 'crafter_ued_results_mask5_no_history.csv')
csv_no_div_path = os.path.join(SCRIPT_DIR, 'logs', 'results', 'crafter_ued_results_mask5_no_diversity.csv')

# -----------------------------
# Helper
# -----------------------------
def _aggregate_seed_curves(
    df, y_cols, seed_col='Seed', iter_col='Iter', warmup_iter=0,
    step_col=None, x_scale=1000, recompute_after_warmup=False
):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    needed = [seed_col, iter_col] + [c for c in y_cols if c in df.columns]
    if step_col and step_col in df.columns:
        needed.append(step_col)
    needed = list(dict.fromkeys(needed))
    df = df[needed].copy()
    df = df.sort_values([seed_col, iter_col])

    per_seed = []
    for seed, g in df.groupby(seed_col):
        g = g.sort_values(iter_col).copy()
        if warmup_iter > 0:
            g = g[pd.to_numeric(g[iter_col], errors='coerce') > warmup_iter].copy()
            if len(g) == 0:
                continue

        if recompute_after_warmup and step_col and step_col in g.columns:
            g['x_raw'] = pd.to_numeric(g[step_col], errors='coerce').cumsum()
        elif step_col and step_col in g.columns:
            g['x_raw'] = pd.to_numeric(g[step_col], errors='coerce').cumsum()
        else:
            g['x_raw'] = pd.to_numeric(g[iter_col], errors='coerce')

        g['x_raw'] = g['x_raw'] - g['x_raw'].iloc[0]
        g['x_k'] = g['x_raw'] / x_scale
        g[seed_col] = seed
        per_seed.append(g)

    if len(per_seed) == 0:
        return None
    merged = pd.concat(per_seed, axis=0, ignore_index=True)

    valid_y_cols = [c for c in y_cols if c in merged.columns]
    if len(valid_y_cols) == 0:
        return None

    for c in valid_y_cols:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    agg_dict = {c: ['mean', 'std'] for c in valid_y_cols}
    out = merged.groupby('x_k', as_index=False).agg(agg_dict)
    out.columns = [
        col[0] if col[1] == '' else (col[0] if col[1] == 'mean' else f'{col[0]}_{col[1]}')
        for col in out.columns.to_flat_index()
    ]
    return out


# -----------------------------
# Plot: MAC vs Ablations (1x2)
# -----------------------------
def run_ablation_plot():
    target_col = 'target_val_avg_val_loss_wm'
    ce_col = 'target_val_val_ce_loss'
    inv_col = 'target_val_val_inv_loss'
    y_cols = [target_col, ce_col, inv_col]

    warmup = 10  # MAC warmup iterations

    # Load data
    datasets = {}
    for name, path in [('MAC', csv_mac_path), ('no_history', csv_no_his_path), ('no_diversity', csv_no_div_path)]:
        if os.path.exists(path):
            df_raw = pd.read_csv(path).drop_duplicates(subset=['Seed', 'Iter'], keep='last')
            datasets[name] = _aggregate_seed_curves(
                df_raw, y_cols=y_cols, seed_col='Seed', iter_col='Iter',
                warmup_iter=warmup, step_col='New_Data_Size', x_scale=1000,
                recompute_after_warmup=True
            )
        else:
            print(f"[Skip] {name}: {path} not found")
            datasets[name] = None

    # Style config
    styles = {
        'MAC':          {'color': '#1F5FBF', 'ls': '-',  'label': 'MAC (Full)',      'lw': 2.2},
        'no_history':   {'color': '#E07B39', 'ls': '--', 'label': 'MAC w/o History', 'lw': 2.0},
        'no_diversity': {'color': '#2CA02C', 'ls': '-.', 'label': 'MAC w/o Diversity', 'lw': 2.0},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    plot_configs = [
        (target_col, r'Target Val Loss ($\mathcal{L}_{total}$)'),
        (ce_col, r'Target Val CE Loss ($\mathcal{L}_{ce}$)'),
    ]

    for ax_idx, (col, ylabel) in enumerate(plot_configs):
        ax = axes[ax_idx]
        for name in ['MAC', 'no_history', 'no_diversity']:
            d = datasets.get(name)
            if d is None or col not in d.columns:
                continue
            s = styles[name]
            x = pd.to_numeric(d['x_k'], errors='coerce').to_numpy(dtype=float)
            y = pd.to_numeric(d[col], errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if mask.sum() == 0:
                continue

            # EMA smoothing
            alpha_ema = 0.35
            y_smooth = pd.Series(y[mask]).ewm(alpha=alpha_ema, adjust=False).mean().to_numpy()

            ax.plot(x[mask], y[mask], color=s['color'], alpha=0.15, linewidth=1.0)
            ax.plot(x[mask], y_smooth, color=s['color'], ls=s['ls'], label=s['label'], linewidth=s['lw'])

            std_col = col + '_std'
            if std_col in d.columns:
                ystd = pd.to_numeric(d[std_col], errors='coerce').fillna(0).to_numpy(dtype=float)[mask]
                ystd_smooth = pd.Series(ystd).ewm(alpha=alpha_ema, adjust=False).mean().to_numpy()
                ax.fill_between(
                    x[mask], y_smooth - ystd_smooth * 0.5, y_smooth + ystd_smooth * 0.5,
                    color=s['color'], alpha=0.1, lw=0
                )

        ax.set_xlabel(r'Transitions ($\times 10^3$)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, which='both', linestyle=':', alpha=0.5, linewidth=0.5)
        ax.legend(frameon=True, edgecolor='0.8', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(csv_mac_path), '..', '..', 'crafter_ablation_plot.png')
    save_path = os.path.abspath(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.show()


if __name__ == '__main__':
    run_ablation_plot()
