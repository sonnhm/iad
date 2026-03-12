import torch
import os


def save_checkpoint(model, optimizer, epoch, path):

    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    file_path = os.path.join(
        path,
        f"epoch_{epoch}.pth"
    )

    torch.save(checkpoint, file_path)

    print(f"Checkpoint saved: {file_path}")