"""
Train CustomResNet18 backbone bằng Knowledge Distillation.

Teacher (pretrained ResNet18) hướng dẫn Student (CustomResNet18) cách trích features.
Sau khi train, backbone được dùng trong PatchCore.

Usage:
    python training/train_backbone.py --category bottle
    python training/train_backbone.py --category bottle --epochs 30 --lr 0.001
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.config import load_config
from data_processing.mvtec import MVTecDataset
from training.backbone_trainer import BackboneTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train PatchCore Backbone (Knowledge Distillation)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    root = data_cfg.get("root", "datasets/mvtec")
    category = args.category or data_cfg.get("category", "bottle")
    image_size = data_cfg.get("image_size", 256)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    epochs = args.epochs or train_cfg.get("backbone_epochs", 30)
    lr = args.lr or train_cfg.get("backbone_lr", 1e-3)
    checkpoint_dir = f"checkpoints/backbone/{category}"

    print(f"\n{'='*60}")
    print(f"Training PatchCore Backbone (Knowledge Distillation)")
    print(f"{'='*60}")
    print(f"  Category: {category}")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Datasets
    train_dataset = MVTecDataset(root, category, split="train", image_size=image_size)
    valid_dataset = MVTecDataset(root, category, split="valid", image_size=image_size)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(valid_dataset)} images")

    # Train
    trainer = BackboneTrainer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.train()
    print(f"\nBackbone saved to: {checkpoint_dir}/best.pth")


if __name__ == "__main__":
    main()
