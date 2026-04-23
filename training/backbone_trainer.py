"""
BackboneTrainer — Knowledge Distillation để train CustomResNet18.

Ý tưởng:
    - Teacher: pretrained ResNet18 (torchvision) — FROZEN, không update weights
    - Student: CustomResNet18 (tự define) — LEARNABLE, cần train
    - Loss: MSE giữa intermediate features (layer1-4) của teacher và student
    - Sau khi train, student đã "học" cách trích features tương tự teacher
    → Student backbone được dùng trong PatchCore

Tại sao Knowledge Distillation?
    - Student tự define kiến trúc (thể hiện khả năng hiểu ResNet)
    - Có quá trình training rõ ràng (loss curves, epochs, checkpoints)
    - Không copy pretrained weights — student phải TỰ HỌC
    - Teacher chỉ dùng lúc train, sau đó bỏ
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from app_utils.checkpoint import save_checkpoint
from app_utils.logger import get_logger
from models.custom_resnet import CustomResNet18

logger = get_logger("backbone_trainer")


class FeatureHook:
    """
    Hook để capture intermediate features từ 1 layer.

    Dùng register_forward_hook để "bắt" output tại layer bất kỳ
    mà không cần sửa forward() của model.
    """

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Callback: lưu output khi forward qua layer."""
        self.features = output

    def remove(self):
        """Gỡ hook."""
        self.hook.remove()


class BackboneTrainer:
    """
    Knowledge Distillation trainer cho CustomResNet18.

    Teacher: torchvision ResNet18 (pretrained ImageNet) — FROZEN
    Student: CustomResNet18 (tự define) — LEARNABLE

    Loss = Σ MSE(student_layer_i, teacher_layer_i) for i = 1..4

    Args:
        train_dataset: training dataset (ảnh normal)
        valid_dataset: validation dataset (ảnh normal)
        batch_size: batch size
        lr: learning rate
        epochs: số epochs
        checkpoint_dir: thư mục lưu checkpoints
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset=None,
        batch_size=32,
        lr=1e-3,
        epochs=30,
        checkpoint_dir="checkpoints/backbone",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # ========== Teacher: pretrained ResNet18 (FROZEN) ==========
        self.teacher = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()

        # Freeze teacher — KHÔNG update weights
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Hook vào 4 layers của teacher
        self.teacher_hooks = [
            FeatureHook(self.teacher.layer1),
            FeatureHook(self.teacher.layer2),
            FeatureHook(self.teacher.layer3),
            FeatureHook(self.teacher.layer4),
        ]

        # ========== Student: CustomResNet18 (LEARNABLE) ==========
        self.student = CustomResNet18(num_classes=None)
        self.student = self.student.to(self.device)

        # Hook vào 4 layers của student
        self.student_hooks = [
            FeatureHook(self.student.layer1),
            FeatureHook(self.student.layer2),
            FeatureHook(self.student.layer3),
            FeatureHook(self.student.layer4),
        ]

        # ========== Training setup ==========
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
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float("inf")

        # Mixed Precision training (AMP) -- reduces VRAM ~30%, speeds training ~1.5x.
        # Only enabled on CUDA; CPU training falls back to FP32 automatically.
        # KD architecture and loss function are unchanged -- only numerical precision
        # of forward/backward passes is affected.
        self._use_amp = self.device.type == "cuda"
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

    def _compute_distillation_loss(self):
        """
        Tính Knowledge Distillation loss.

        Loss = Σ MSE(student_layer_i, teacher_layer_i) cho i = 1..4

        Mỗi layer có kích thước khác nhau:
            layer1: (B, 64,  H/4,  W/4)
            layer2: (B, 128, H/8,  W/8)
            layer3: (B, 256, H/16, W/16)
            layer4: (B, 512, H/32, W/32)

        Returns:
            total_loss: scalar tensor
            layer_losses: list of 4 scalar values (cho logging)
        """
        total_loss = 0.0
        layer_losses = []

        for t_hook, s_hook in zip(self.teacher_hooks, self.student_hooks):
            t_feat = t_hook.features.detach()  # Detach teacher (no grad)
            s_feat = s_hook.features

            loss = self.criterion(s_feat, t_feat)
            total_loss += loss
            layer_losses.append(loss.item())

        return total_loss, layer_losses

    def train(self):
        """Chạy Knowledge Distillation training loop."""
        logger.info(f"Starting Knowledge Distillation: {self.epochs} epochs")
        logger.info(f"  Teacher: ResNet18 (pretrained, frozen)")
        logger.info(f"  Student: CustomResNet18 (learnable)")

        import json
        import os

        history = []

        for epoch in range(self.epochs):
            # --- Training ---
            train_loss, train_layer_losses = self._train_one_epoch()

            # --- Validation ---
            val_loss = None
            if self.valid_loader is not None:
                val_loss = self._validate()

            # --- Logging ---
            layer_str = " | ".join(
                [f"L{i+1}={l:.4f}" for i, l in enumerate(train_layer_losses)]
            )
            msg = f"Epoch {epoch + 1}/{self.epochs} | Loss: {train_loss:.6f} | {layer_str}"
            if val_loss is not None:
                msg += f" | Val: {val_loss:.6f}"
            logger.info(msg)

            # --- Save checkpoint ---
            save_checkpoint(self.student, self.optimizer, epoch, self.checkpoint_dir)

            # --- Save best model ---
            compare_loss = val_loss if val_loss is not None else train_loss
            if compare_loss < self.best_val_loss:
                self.best_val_loss = compare_loss
                save_checkpoint(
                    self.student,
                    self.optimizer,
                    epoch,
                    self.checkpoint_dir,
                    filename="best.pth",
                )
                logger.info(f"  → New best model (loss={compare_loss:.6f})")

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

        logger.info("Knowledge Distillation completed!")

        # Cleanup hooks
        for h in self.teacher_hooks + self.student_hooks:
            h.remove()

    def _train_one_epoch(self):
        """
        Train 1 epoch with Mixed Precision (AMP) when CUDA is available.

        AMP uses FP16 for forward pass (student) and loss computation,
        while GradScaler ensures gradient stability during backprop.
        Teacher forward remains FP32 (frozen, no-grad -- not affected by AMP).
        KD loss (Sigma MSE layer1..4) is structurally unchanged.
        """
        self.student.train()
        total_loss = 0.0
        total_layer_losses = [0.0, 0.0, 0.0, 0.0]

        for images, _ in self.train_loader:
            images = images.to(self.device)

            # Teacher forward: frozen, no grad, stays FP32
            with torch.no_grad():
                self.teacher(images)  # Triggers teacher hooks

            # Student forward + KD loss under autocast (FP16 on CUDA)
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                self.student(images)  # Triggers student hooks
                loss, layer_losses = self._compute_distillation_loss()

            # AMP-aware backward: scaler handles gradient unscaling
            self.optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()
            for i, ll in enumerate(layer_losses):
                total_layer_losses[i] += ll

        n = len(self.train_loader)
        avg_layer = [l / n for l in total_layer_losses]
        return total_loss / n, avg_layer

    def _validate(self):
        """Validate, trả về average loss."""
        self.student.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, _ in self.valid_loader:
                images = images.to(self.device)

                self.teacher(images)
                self.student(images)

                loss, _ = self._compute_distillation_loss()
                total_loss += loss.item()

        return total_loss / len(self.valid_loader)
