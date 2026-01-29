
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

def plot_ablation_comparison(full_ued_path, ablation_path, output_dir="trainer/logs/results"):
    """
    Reads two CSVs (Full UED vs Ablation) and plots comparison metrics.
    """
    
    # 1. Check files
    if not os.path.exists(full_ued_path):
        print(f"[Error] File not found: {full_ued_path}")
        return
    if not os.path.exists(ablation_path):
        print(f"[Error] File not found: {ablation_path}")
        return

    # 2. Load Data
    df_ued = pd.read_csv(full_ued_path)
    df_ablation = pd.read_csv(ablation_path)

    # 3. Define Metrics to Compare
    # (Metric Column, Display Name)
    metrics = [
        ("WM_Val_Loss", "WM Validation Loss (Lower is Better)"),
        ("Gen_Div_Reward", "Generator Diversity (Higher is Better)"),
        ("Gen_Real_Loss", "Map Difficulty (WM Prediction Error)"),
        ("Solvable_Count", "Solvable Maps Count (Max 3)")
    ]

    # 4. Create Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Ablation Study: Full UED vs w/o Diversity", fontsize=16)
    
    for i, (col, title) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # Plot UED
        if col in df_ued.columns:
            ax.plot(df_ued["Iter"], df_ued[col], label="Full UED", marker="o", linestyle="-", color="blue", alpha=0.7)
        
        # Plot Ablation
        if col in df_ablation.columns:
            ax.plot(df_ablation["Iter"], df_ablation[col], label="w/o Diversity", marker="x", linestyle="--", color="red", alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        
        # Log scale for Loss
        if "Loss" in col:
            ax.set_yscale("log")

    plt.tight_layout()
    
    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ablation_comparison_diversity.png")
    plt.savefig(save_path)
    print(f"[Success] Comparison plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Default Paths assumed relative to where script is run (usually from root)
    base_dir = "trainer/logs/results"
    
    # Allow command line args
    if len(sys.argv) > 2:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        file1 = os.path.join(base_dir, "experiment_summary_mask3_mse.csv")
        file2 = os.path.join(base_dir, "experiment_summary_mask3_mse_no_diversity.csv")
        
    print(f"Comparing:")
    print(f"1. {file1}")
    print(f"2. {file2}")
    
    plot_ablation_comparison(file1, file2)
