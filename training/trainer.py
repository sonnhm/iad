import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.checkpoint import save_checkpoint


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        batch_size=16,
        lr=1e-3,
        epochs=10,
        checkpoint_dir="checkpoints"
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device:", self.device)

        torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,   # đổi 4 -> 0
            pin_memory=True
    )

        self.epochs = epochs

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.checkpoint_dir = checkpoint_dir


    def train(self):

        for epoch in range(self.epochs):

            total_loss = 0

            for images, _ in self.dataloader:

                images = images.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, images)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)

            print(f"Epoch {epoch} | Loss {avg_loss:.4f}")

            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.checkpoint_dir
            )