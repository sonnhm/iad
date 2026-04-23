"""
Evaluate Autoencoder trên MVTec AD.

Usage:
    python experiments/run_autoencoder.py --category bottle
    python experiments/run_autoencoder.py --category cable --checkpoint path/to/best.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app_utils.checkpoint import load_checkpoint
from data_processing.mvtec import MVTecDataset
from evaluation.metrics import anomaly_score, evaluate_all, print_metrics
from models.autoencoder import Autoencoder

DATA_ROOT = "datasets/mvtec"


def run(category, checkpoint_path=None):
    """
    Evaluate Autoencoder trên 1 category.

    Args:
        category: MVTec category name
        checkpoint_path: đường dẫn tới checkpoint (auto nếu None)

    Returns:
        metrics: dict chứa tất cả evaluation metrics
    """
    # Auto checkpoint path per category
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/autoencoder/{category}/best.pth"

    if not os.path.exists(checkpoint_path):
        print(f"  [WARNING] Checkpoint not found: {checkpoint_path}")
        print(f"  → Cần train trước: python training/train.py --category {category}")
        return None

    # Load test dataset
    dataset = MVTecDataset(root=DATA_ROOT, category=category, split="test")

    # Load model
    model = Autoencoder()
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # Inference
    scores = []
    labels = []

    for img, label in dataset:
        img = img.unsqueeze(0)

        with torch.no_grad():
            recon = model(img)

        score = anomaly_score(img, recon)
        scores.append(score.item())
        labels.append(label)

    # Tính metrics
    metrics = evaluate_all(labels, scores)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Autoencoder")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    metrics = run(args.category, args.checkpoint)
    if metrics:
        print_metrics(metrics, model_name=f"Autoencoder ({args.category})")
