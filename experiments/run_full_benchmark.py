"""
Full benchmark — chạy toàn bộ pipeline từ training đến evaluation.

Pipeline cho mỗi category:
    1. Train backbone (Knowledge Distillation)  → checkpoints/backbone/{cat}/best.pth
    2. Train Autoencoder                         → checkpoints/autoencoder/{cat}/best.pth
    3. Evaluate PatchCore (trained backbone)
    4. Evaluate Autoencoder (trained)
    5. Evaluate CNN+OC-SVM (pretrained features)

Usage:
    python experiments/run_full_benchmark.py --category bottle
    python experiments/run_full_benchmark.py --category all
    python experiments/run_full_benchmark.py --category all --skip-training
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.config import ALL_CATEGORIES
from data_processing.mvtec import MVTecDataset
from evaluation.metrics import print_metrics
from experiments.run_autoencoder import run as run_autoencoder
from experiments.run_ocsvm import run as run_ocsvm
from experiments.run_patchcore import run as run_patchcore
from models.autoencoder import Autoencoder
from training.backbone_trainer import BackboneTrainer
from training.trainer import Trainer


def run_benchmark(
    category,
    backbone_epochs=30,
    ae_epochs=50,
    batch_size=32,
    lr=1e-3,
    skip_training=False,
):
    """
    Chạy full benchmark cho 1 category.

    Args:
        category: MVTec category name
        backbone_epochs: epochs cho Knowledge Distillation
        ae_epochs: epochs cho Autoencoder
        batch_size: batch size
        lr: learning rate
        skip_training: True = chỉ evaluate (đã train trước)
    """
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {category}")
    print(f"{'='*60}")

    backbone_ckpt_dir = f"checkpoints/backbone/{category}"
    ae_ckpt_dir = f"checkpoints/autoencoder/{category}"

    if not skip_training:
        # === Step 1: Train Backbone (Knowledge Distillation) ===
        print(f"\n  [1/5] Training PatchCore Backbone (KD)...")
        train_dataset = MVTecDataset("datasets/mvtec", category, split="train")
        valid_dataset = MVTecDataset("datasets/mvtec", category, split="valid")

        bb_trainer = BackboneTrainer(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size=batch_size,
            lr=lr,
            epochs=backbone_epochs,
            checkpoint_dir=backbone_ckpt_dir,
        )
        bb_trainer.train()

        # === Step 2: Train Autoencoder ===
        print(f"\n  [2/5] Training Autoencoder...")
        train_dataset = MVTecDataset("datasets/mvtec", category, split="train")
        valid_dataset = MVTecDataset("datasets/mvtec", category, split="valid")

        model = Autoencoder()
        ae_trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size=batch_size,
            epochs=ae_epochs,
            lr=lr,
            checkpoint_dir=ae_ckpt_dir,
        )
        ae_trainer.train()
    else:
        print(f"\n  [SKIP] Training skipped (--skip-training)")

    # === Step 3: Evaluate PatchCore ===
    print(f"\n  [3/5] Evaluating PatchCore...")
    pc_metrics = run_patchcore(category)
    print_metrics(pc_metrics, f"PatchCore ({category})")

    # === Step 4: Evaluate Autoencoder ===
    print(f"\n  [4/5] Evaluating Autoencoder...")
    ae_metrics = run_autoencoder(category)
    if ae_metrics:
        print_metrics(ae_metrics, f"Autoencoder ({category})")

    # === Step 5: Evaluate CNN+OC-SVM ===
    print(f"\n  [5/5] Evaluating CNN+OC-SVM...")
    oc_metrics = run_ocsvm(category)
    print_metrics(oc_metrics, f"CNN+OC-SVM ({category})")

    # === Results summary ===
    print(f"\n{'='*60}")
    print(f"  RESULTS for {category}:")
    print(f"  PatchCore:   AUROC={pc_metrics['auroc']:.4f} | F1={pc_metrics['f1']:.4f}")
    if ae_metrics:
        print(
            f"  Autoencoder: AUROC={ae_metrics['auroc']:.4f} | F1={ae_metrics['f1']:.4f}"
        )
    else:
        print(f"  Autoencoder: N/A (checkpoint missing)")
    print(f"  CNN+OC-SVM:  AUROC={oc_metrics['auroc']:.4f} | F1={oc_metrics['f1']:.4f}")
    print(f"{'='*60}")

    # Lưu results per category
    os.makedirs("results", exist_ok=True)
    result = {
        "category": category,
        "patchcore": pc_metrics,
        "autoencoder": ae_metrics,
        "cnn_ocsvm": oc_metrics,
    }
    with open(f"results/{category}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Benchmark")
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        help="Category name or 'all' for all 15 categories",
    )
    parser.add_argument("--backbone-epochs", type=int, default=30)
    parser.add_argument("--ae-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training, only evaluate"
    )
    args = parser.parse_args()

    if args.category == "all":
        categories = ALL_CATEGORIES
    else:
        categories = [args.category]

    all_results = []
    for cat in categories:
        result = run_benchmark(
            cat,
            backbone_epochs=args.backbone_epochs,
            ae_epochs=args.ae_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            skip_training=args.skip_training,
        )
        all_results.append(result)

    # Final summary
    if len(categories) > 1:
        print(f"\n{'='*70}")
        print(f"  ALL CATEGORIES BENCHMARK COMPLETE")
        print(f"{'='*70}")
        for r in all_results:
            cat = r["category"]
            pc = r["patchcore"]["auroc"]
            ae = r["autoencoder"]["auroc"] if r["autoencoder"] else "N/A"
            oc = r["cnn_ocsvm"]["auroc"]
            print(
                f"  {cat:15s} | PC={pc:.4f} | AE={ae if isinstance(ae, str) else f'{ae:.4f}'} | OC={oc:.4f}"
            )
        print(f"{'='*70}")
