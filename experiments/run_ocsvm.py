"""
Evaluate CNN + One-Class SVM trên MVTec AD.

Pipeline:
    1. Trích feature vectors từ ResNet18 pretrained (training data)
    2. Train One-Class SVM trên features bình thường
    3. Test: trích features → OC-SVM decision function → anomaly score

Usage:
    python experiments/run_ocsvm.py --category bottle
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader

from data_processing.mvtec import MVTecDataset
from evaluation.metrics import evaluate_all, print_metrics
from models.cnn_feature import CNNFeatureExtractor

DATA_ROOT = "datasets/mvtec"


def run(category, batch_size=32, gamma="auto"):
    """
    Evaluate CNN + OC-SVM trên 1 category.

    Args:
        category: MVTec category name
        batch_size: batch size cho feature extraction
        gamma: SVM kernel parameter

    Returns:
        metrics: dict chứa tất cả evaluation metrics
    """
    # Datasets
    train_dataset = MVTecDataset(root=DATA_ROOT, category=category, split="train")
    test_dataset = MVTecDataset(root=DATA_ROOT, category=category, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Feature extractor (pretrained ResNet18)
    model = CNNFeatureExtractor()
    model.eval()

    # --- Trích training features ---
    train_features = []
    for imgs, _ in train_loader:
        with torch.no_grad():
            feats = model(imgs)
        train_features.append(feats.numpy())

    train_features = np.concatenate(train_features, axis=0)
    print(f"  Train features shape: {train_features.shape}")

    # --- Train One-Class SVM ---
    ocsvm = OneClassSVM(gamma=gamma)
    ocsvm.fit(train_features)

    # --- Inference ---
    scores = []
    labels = []

    for imgs, labels_batch in test_loader:
        with torch.no_grad():
            feats = model(imgs)

        feats = feats.numpy()

        # Đảo dấu: score cao = anomaly
        score = -ocsvm.decision_function(feats)
        scores.extend(score.tolist())
        labels.extend(labels_batch.tolist())

    # Tính metrics
    metrics = evaluate_all(labels, scores)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN + OC-SVM")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    metrics = run(args.category, args.batch_size)
    if metrics:
        print_metrics(metrics, model_name=f"CNN+OC-SVM ({args.category})")
