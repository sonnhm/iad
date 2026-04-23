"""
Trainer — training loop cho Autoencoder.

Bao gồm:
- Training loop với MSE loss
- Validation step mỗi epoch
- Lưu best model dựa trên validation loss
- Checkpoint mỗi epoch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app_utils.checkpoint import save_checkpoint
from app_utils.logger import get_logger

logger = get_logger("trainer")


class Trainer:
    """
    Trainer cho Autoencoder anomaly detection.

    Args:
        model: Autoencoder model
        train_dataset: training dataset (ảnh bình thường)
        valid_dataset: validation dataset (ảnh bình thường)
        batch_size: batch size
        lr: learning rate
        epochs: số epochs
        checkpoint_dir: thư mục lưu checkpoints
    """

    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset=None,
        batch_size=32,
        lr=1e-3,
        epochs=50,
        checkpoint_dir="checkpoints/autoencoder",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.model = model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        self.valid_loader = None
        if valid_dataset is not None and len(valid_dataset) > 0:
            self.valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir

        self.best_val_loss = float("inf")

    def train(self):
        """Chạy training loop."""
        import json
        import os

        history = []

        for epoch in range(self.epochs):
            # --- Training ---
            train_loss = self._train_one_epoch()

            # --- Validation ---
            val_loss = None
            if self.valid_loader is not None:
                val_loss = self._validate()

            # --- Logging ---
            msg = f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.6f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.6f}"
            logger.info(msg)

            # --- Save checkpoint ---
            save_checkpoint(self.model, self.optimizer, epoch, self.checkpoint_dir)

            # --- Save best model ---
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.checkpoint_dir,
                    filename="best.pth",
                )
                logger.info(f"  → New best model (val_loss={val_loss:.6f})")

            # --- Save history ---
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loss is not None else train_loss,
                }
            )
            try:
                history_path = os.path.join(self.checkpoint_dir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=4)
            except Exception as e:
                logger.warning(f"Failed to save history: {e}")

        logger.info("Training completed!")

    def _train_one_epoch(self):
        """Train 1 epoch, trả về average loss."""
        self.model.train()
        total_loss = 0.0

        for images, _ in self.train_loader:
            images = images.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, images)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        """Validate, trả về average loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, _ in self.valid_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                total_loss += loss.item()

        return total_loss / len(self.valid_loader)
