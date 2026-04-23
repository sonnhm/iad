"""
Train Autoencoder trên MVTec AD.

Sử dụng config YAML hoặc CLI args.
Checkpoint tự động lưu theo category: checkpoints/autoencoder/{category}/

Usage:
    python training/train.py --category bottle
    python training/train.py --category bottle --epochs 50
    python training/train.py --config configs/autoencoder.yaml --category cable
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.config import load_config
from data_processing.mvtec import MVTecDataset
from models.autoencoder import Autoencoder
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
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
    category = args.category  # CLI always takes priority
    image_size = data_cfg.get("image_size", 256)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    epochs = args.epochs or train_cfg.get("epochs", 50)
    lr = args.lr or train_cfg.get("lr", 1e-3)

    # Checkpoint dir per category
    checkpoint_dir = f"checkpoints/autoencoder/{category}"

    print(f"\n{'='*60}")
    print(f"Training Autoencoder — {category}")
    print(f"{'='*60}")
    print(f"  Image size: {image_size}, Batch size: {batch_size}")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Datasets
    train_dataset = MVTecDataset(root, category, split="train", image_size=image_size)
    valid_dataset = MVTecDataset(root, category, split="valid", image_size=image_size)

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(valid_dataset)} images")
    print(f"{'='*60}\n")

    # Model + Train
    model = Autoencoder()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train()

    print(f"\nAutoencoder saved to: {checkpoint_dir}/best.pth")


if __name__ == "__main__":
    main()
