"""
So sánh 3 models trên tất cả 15 MVTec AD categories.

Chạy Autoencoder, CNN+OC-SVM, PatchCore → lưu results.csv với tất cả metrics.

Yêu cầu:
    - Đã train backbone cho PatchCore: python training/train_backbone.py --category {cat}
    - Đã train Autoencoder: python training/train.py --category {cat}

Usage:
    python experiments/compare_models.py
    python experiments/compare_models.py --categories bottle cable
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.config import ALL_CATEGORIES
from evaluation.metrics import print_metrics
from experiments.run_autoencoder import run as run_autoencoder
from experiments.run_ocsvm import run as run_ocsvm
from experiments.run_patchcore import run as run_patchcore


def main():
    parser = argparse.ArgumentParser(description="Compare 3 Models")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to evaluate (default: all 15)",
    )
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    categories = args.categories or ALL_CATEGORIES
    results = []

    for category in categories:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {category}")
        print(f"{'='*60}")

        # --- Autoencoder ---
        print(f"\n  [1/3] Autoencoder...")
        ae_metrics = run_autoencoder(category)
        if ae_metrics:
            print_metrics(ae_metrics, f"Autoencoder ({category})")

        # --- CNN+OC-SVM ---
        print(f"\n  [2/3] CNN+OC-SVM...")
        oc_metrics = run_ocsvm(category)
        if oc_metrics:
            print_metrics(oc_metrics, f"CNN+OC-SVM ({category})")

        # --- PatchCore ---
        print(f"\n  [3/3] PatchCore...")
        pc_metrics = run_patchcore(category)
        if pc_metrics:
            print_metrics(pc_metrics, f"PatchCore ({category})")

        results.append(
            {
                "category": category,
                "ae": ae_metrics,
                "oc": oc_metrics,
                "pc": pc_metrics,
            }
        )

    # ========== Lưu CSV ==========
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "Category",
                "AE_AUROC",
                "AE_AP",
                "AE_F1",
                "AE_Precision",
                "AE_Recall",
                "AE_Specificity",
                "OCSVM_AUROC",
                "OCSVM_AP",
                "OCSVM_F1",
                "OCSVM_Precision",
                "OCSVM_Recall",
                "OCSVM_Specificity",
                "PC_AUROC",
                "PC_AP",
                "PC_F1",
                "PC_Precision",
                "PC_Recall",
                "PC_Specificity",
                "PC_Pixel_AUROC",
                "PC_PRO",
            ]
        )

        # Data rows
        for r in results:
            row = [r["category"]]

            for model_key in ["ae", "oc", "pc"]:
                m = r[model_key]
                if m is not None:
                    row.extend(
                        [
                            f"{m['auroc']:.4f}",
                            f"{m['average_precision']:.4f}",
                            f"{m['f1']:.4f}",
                            f"{m['precision']:.4f}",
                            f"{m['recall']:.4f}",
                            f"{m['specificity']:.4f}",
                        ]
                    )
                else:
                    row.extend(["N/A"] * 6)

            # Pixel-level metrics (chỉ PatchCore)
            pc = r["pc"]
            if pc is not None:
                row.append(f"{pc.get('pixel_auroc', 0):.4f}")
                row.append(f"{pc.get('pro_score', 0):.4f}")
            else:
                row.extend(["N/A"] * 2)

            writer.writerow(row)

    print(f"\nResults saved to {args.output}")

    # ========== Lưu Optimal Thresholds ra JSON ==========
    import json

    thresholds_file = "checkpoints/thresholds.json"
    os.makedirs(os.path.dirname(thresholds_file), exist_ok=True)

    thresholds_data = {}
    if os.path.exists(thresholds_file):
        try:
            with open(thresholds_file, "r") as f:
                thresholds_data = json.load(f)
        except Exception:
            pass

    for r in results:
        cat = r["category"]
        if cat not in thresholds_data:
            thresholds_data[cat] = {}
        if r["ae"]:
            thresholds_data[cat]["autoencoder"] = r["ae"]["threshold"]
        if r["oc"]:
            thresholds_data[cat]["cnn_ocsvm"] = r["oc"]["threshold"]
        if r["pc"]:
            thresholds_data[cat]["patchcore"] = r["pc"]["threshold"]

    with open(thresholds_file, "w") as f:
        json.dump(thresholds_data, f, indent=4)

    print(f"Optimal Thresholds saved to {thresholds_file}")

    # ========== In summary ==========
    print(f"\n{'='*120}")
    print("SUMMARY — Tổng hợp kết quả tất cả models")
    print(f"{'='*120}")

    # Header chú thích
    print("\n Chú thích metrics:")
    print("  AUROC      = Khả năng xếp hạng (1.0 = hoàn hảo, 0.5 = random)")
    print("  AP         = Average Precision (tốt cho data mất cân bằng)")
    print("  F1         = Trung bình hài hòa Precision & Recall")
    print("  Prec       = Precision — bao nhiêu % dự đoán anomaly là đúng")
    print("  Rec        = Recall — phát hiện được bao nhiêu % anomaly thực tế")
    print("  Spec       = Specificity — nhận đúng bao nhiêu % ảnh bình thường")

    # Bảng kết quả
    header = (
        f"\n{'Category':15s} | "
        f"{'--- Autoencoder ---':^42s} | "
        f"{'--- CNN+OC-SVM ---':^42s} | "
        f"{'--- PatchCore ---':^42s}"
    )
    sub_header = (
        f"{'':15s} | "
        f"{'AUROC':>6s} {'AP':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} | "
        f"{'AUROC':>6s} {'AP':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s} | "
        f"{'AUROC':>6s} {'AP':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Spec':>6s}"
    )
    print(header)
    print(sub_header)
    print(f"{'-'*15}-+-{'-'*42}-+-{'-'*42}-+-{'-'*42}")

    for r in results:
        parts = [f"{r['category']:15s}"]
        for key in ["ae", "oc", "pc"]:
            m = r[key]
            if m:
                parts.append(
                    f"{m['auroc']:6.3f} {m['average_precision']:6.3f} "
                    f"{m['f1']:6.3f} {m['precision']:6.3f} "
                    f"{m['recall']:6.3f} {m['specificity']:6.3f}"
                )
            else:
                parts.append(f"{'N/A':^42s}")
        print(" | ".join(parts))

    print(f"{'='*120}")


if __name__ == "__main__":
    main()
