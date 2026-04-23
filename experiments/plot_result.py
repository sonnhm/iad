"""
Plot kết quả so sánh models từ results.csv.

Hỗ trợ cả format CSV cũ (Autoencoder, OCSVM, PatchCore columns)
và format mới (AE_AUROC, OCSVM_AUROC, PC_AUROC columns).

Usage:
    python experiments/plot_result.py
    python experiments/plot_result.py --csv results.csv --output model_comparison.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(csv_path="results.csv", output_path="model_comparison.png"):
    """
    Vẽ biểu đồ so sánh AUROC, AP, F1 của 3 models.

    Args:
        csv_path: đường dẫn file CSV kết quả
        output_path: đường dẫn lưu ảnh output
    """
    df = pd.read_csv(csv_path)
    categories = df["Category"]
    x = np.arange(len(categories))
    width = 0.25

    # Detect column format
    if "AE_AUROC" in df.columns:
        # New format
        ae_auroc = pd.to_numeric(df["AE_AUROC"], errors="coerce")
        oc_auroc = pd.to_numeric(df["OCSVM_AUROC"], errors="coerce")
        pc_auroc = pd.to_numeric(df["PC_AUROC"], errors="coerce")
    elif "Autoencoder" in df.columns:
        # Old format
        ae_auroc = pd.to_numeric(df["Autoencoder"], errors="coerce")
        oc_auroc = pd.to_numeric(df["OCSVM"], errors="coerce")
        pc_auroc = pd.to_numeric(df["PatchCore"], errors="coerce")
    else:
        print("ERROR: Unrecognized CSV format")
        print(f"Columns: {list(df.columns)}")
        return

    # Fill NaN
    ae_auroc = ae_auroc.fillna(0)
    oc_auroc = oc_auroc.fillna(0)
    pc_auroc = pc_auroc.fillna(0)

    # ===== Plot 1: AUROC Bar Chart =====
    fig, ax = plt.subplots(figsize=(16, 7))

    bars1 = ax.bar(
        x - width, ae_auroc, width, label="Autoencoder", color="#6366f1", alpha=0.9
    )
    bars2 = ax.bar(x, oc_auroc, width, label="CNN+OC-SVM", color="#22c55e", alpha=0.9)
    bars3 = ax.bar(
        x + width, pc_auroc, width, label="PatchCore", color="#ef4444", alpha=0.9
    )

    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUROC", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Comparison — AUROC on MVTec AD (15 Categories)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="gray",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {output_path}")
    plt.show()

    # ===== Print Summary =====
    print(f"\n{'='*50}")
    print("Average AUROC:")
    print(f"  Autoencoder: {ae_auroc.mean():.4f}")
    print(f"  CNN+OC-SVM:  {oc_auroc.mean():.4f}")
    print(f"  PatchCore:   {pc_auroc.mean():.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Model Comparison")
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--output", type=str, default="model_comparison.png")
    args = parser.parse_args()

    plot_results(args.csv, args.output)
