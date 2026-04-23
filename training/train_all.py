"""
Train All — chạy toàn bộ training pipeline cho 1 hoặc tất cả categories.

Thứ tự:
    1. Train backbone (Knowledge Distillation)  → checkpoints/backbone/{cat}/best.pth
    2. Train Autoencoder                         → checkpoints/autoencoder/{cat}/best.pth

Sau khi train xong, chạy:
    python experiments/compare_models.py

Usage:
    python training/train_all.py --category bottle
    python training/train_all.py --category all
    python training/train_all.py --category all --backbone-epochs 30 --ae-epochs 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.config import ALL_CATEGORIES
from data_processing.mvtec import MVTecDataset
from models.autoencoder import Autoencoder
from training.backbone_trainer import BackboneTrainer
from training.trainer import Trainer


def train_category(category, backbone_epochs=30, ae_epochs=50, batch_size=32, lr=1e-3):
    """
    Train tất cả models cho 1 category.

    Tự động bỏ qua nếu checkpoint đã tồn tại (tiện khi bị ngắt session,
    chạy lại --category all sẽ chỉ train các category chưa xong).

    Args:
        category: MVTec category name
        backbone_epochs: epochs cho Knowledge Distillation
        ae_epochs: epochs cho Autoencoder
        batch_size: batch size
        lr: learning rate
    """
    backbone_ckpt = f"checkpoints/backbone/{category}"
    ae_ckpt = f"checkpoints/autoencoder/{category}"

    # Auto-skip: cả 2 model đã train xong → bỏ qua category này
    backbone_done = os.path.exists(os.path.join(backbone_ckpt, "best.pth"))
    ae_done = os.path.exists(os.path.join(ae_ckpt, "best.pth"))

    if backbone_done and ae_done:
        print(f"\n   {category} — Đã có đủ checkpoint, bỏ qua!")
        return

    print(f"\n{'='*60}")
    print(f"  TRAINING: {category}")
    print(f"{'='*60}")

    # Load datasets
    train_dataset = MVTecDataset("datasets/mvtec", category, split="train")
    valid_dataset = MVTecDataset("datasets/mvtec", category, split="valid")

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(valid_dataset)} images")

    # === Step 1: Train Backbone (Knowledge Distillation) ===
    if backbone_done:
        print(f"\n  [1/2] Backbone — đã có checkpoint, bỏ qua!")
    else:
        print(f"\n  [1/2] Training PatchCore Backbone...")
        bb_trainer = BackboneTrainer(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            batch_size=batch_size,
            lr=lr,
            epochs=backbone_epochs,
            checkpoint_dir=backbone_ckpt,
        )
        bb_trainer.train()
        print(f"  → Backbone saved: {backbone_ckpt}/best.pth")

    # === Step 2: Train Autoencoder ===
    if ae_done:
        print(f"\n  [2/2] Autoencoder — đã có checkpoint, bỏ qua!")
    else:
        print(f"\n  [2/2] Training Autoencoder...")

        # Reload datasets (trainer consumes them)
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
            checkpoint_dir=ae_ckpt,
        )
        ae_trainer.train()
        print(f"  → Autoencoder saved: {ae_ckpt}/best.pth")

    print(f"\n   {category} — Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train All Models")
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
    args = parser.parse_args()

    if args.category == "all":
        categories = ALL_CATEGORIES
    else:
        categories = [args.category]

    print(f"\n{'#'*60}")
    print(f"  TRAIN ALL — {len(categories)} categories")
    print(f"  Backbone epochs: {args.backbone_epochs}")
    print(f"  Autoencoder epochs: {args.ae_epochs}")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"{'#'*60}")

    for i, cat in enumerate(categories):
        print(f"\n  >>> Category {i+1}/{len(categories)}: {cat}")
        train_category(
            cat,
            backbone_epochs=args.backbone_epochs,
            ae_epochs=args.ae_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    print(f"\n{'#'*60}")
    print(f"  ALL TRAINING COMPLETE!")
    print(f"  Next step: python experiments/compare_models.py")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
