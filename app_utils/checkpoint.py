"""
Checkpoint utilities — lưu và load model checkpoints.
"""

import os

import torch


def save_checkpoint(model, optimizer, epoch, path, filename=None):
    """
    Lưu model checkpoint.

    Args:
        model: PyTorch model
        optimizer: optimizer
        epoch: epoch hiện tại
        path: thư mục lưu checkpoint
        filename: tên file (mặc định: epoch_{epoch}.pth)
    """
    os.makedirs(path, exist_ok=True)

    if filename is None:
        filename = f"epoch_{epoch}.pth"

    filepath = os.path.join(path, filename)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, path, optimizer=None):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model (sẽ được load state_dict vào)
        path: đường dẫn tới file .pth
        optimizer: (optional) optimizer để restore state

    Returns:
        epoch: epoch đã train tới
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded: {path} (epoch {epoch})")

    return epoch
