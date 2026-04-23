"""
Evaluate PatchCore trên MVTec AD.

Pipeline:
    1. Load trained backbone (Knowledge Distillation)
    2. Trích patch features từ training images → memory bank
    3. Coreset subsampling → giảm memory bank
    4. Test: k-NN distance → anomaly score + anomaly map
    5. Pixel-level evaluation: so sánh anomaly map với ground truth mask

Toàn bộ tự implement, KHÔNG dùng sklearn.

Usage:
    python experiments/run_patchcore.py --category bottle
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data_processing.mvtec import MVTecDataset
from evaluation.metrics import evaluate_all, print_metrics
from models.patchcore import PatchCore, PatchCoreFeatureExtractor

DATA_ROOT = "datasets/mvtec"


def run(category, backbone_checkpoint=None, coreset_ratio=0.1, k_neighbors=1):
    """
    Evaluate PatchCore trên 1 category.

    Bao gồm cả image-level và pixel-level metrics (sử dụng ground truth masks).

    Args:
        category: MVTec category name
        backbone_checkpoint: path tới trained backbone (None = auto)
        coreset_ratio: tỷ lệ coreset subsampling
        k_neighbors: số neighbors cho k-NN

    Returns:
        metrics: dict chứa tất cả evaluation metrics (image + pixel level)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto backbone checkpoint path
    if backbone_checkpoint is None:
        backbone_checkpoint = f"checkpoints/backbone/{category}/best.pth"

    # Load backbone
    backbone = PatchCoreFeatureExtractor()

    if os.path.exists(backbone_checkpoint):
        backbone.load_backbone_weights(backbone_checkpoint)
        print(f"  Backbone loaded: {backbone_checkpoint}")
    else:
        print(f"  [WARNING] Backbone checkpoint not found: {backbone_checkpoint}")
        print(
            f"  → Dùng random init. Train trước: python training/train_backbone.py --category {category}"
        )

    # PatchCore
    patchcore = PatchCore(
        backbone=backbone,
        device=device,
        coreset_ratio=coreset_ratio,
        k_neighbors=k_neighbors,
    )

    # Path tới memory bank
    pc_ckpt_dir = f"checkpoints/patchcore/{category}"
    pc_ckpt_path = os.path.join(pc_ckpt_dir, "memory_bank.pth")

    # Nếu đã có memory bank → Load cho nhanh
    if os.path.exists(pc_ckpt_path):
        print(f"  Loading existing Memory Bank: {pc_ckpt_path}")
        patchcore.load(pc_ckpt_path)
    else:
        # Nếu chưa có → Fit & Save
        print(f"  Memory Bank not found. Fitting from training data...")
        train_dataset = MVTecDataset(root=DATA_ROOT, category=category, split="train")
        patchcore.fit(train_dataset)
        os.makedirs(pc_ckpt_dir, exist_ok=True)
        patchcore.save(pc_ckpt_path)

    # Load test dataset VỚI ground truth masks
    test_dataset = MVTecDataset(
        root=DATA_ROOT, category=category, split="test", load_masks=True, mask_size=32
    )

    scores = []
    labels = []
    all_masks = []
    all_anomaly_maps = []

    for item in test_dataset:
        if len(item) == 3:
            img, label, mask = item
        else:
            img, label = item
            mask = None

        score, anomaly_map = patchcore.predict(img)
        scores.append(score[0])
        labels.append(label)

        if mask is not None:
            all_masks.append(mask)
            all_anomaly_maps.append(anomaly_map[0])  # (H', W')

    # Metrics (image-level + pixel-level nếu có masks)
    adaptive_thresh = patchcore.adaptive_threshold
    if all_masks:
        metrics = evaluate_all(
            labels,
            scores,
            masks=all_masks,
            anomaly_maps=all_anomaly_maps,
            adaptive_threshold=adaptive_thresh,
        )
    else:
        metrics = evaluate_all(labels, scores, adaptive_threshold=adaptive_thresh)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PatchCore")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--k-neighbors", type=int, default=1)
    args = parser.parse_args()

    metrics = run(
        args.category,
        args.backbone_checkpoint,
        args.coreset_ratio,
        args.k_neighbors,
    )
    print_metrics(metrics, model_name=f"PatchCore ({args.category})")
