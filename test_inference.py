import torch

from datasets.mvtec import MVTecDataset
from models.autoencoder import Autoencoder
from evaluation.metrics import reconstruction_error


dataset = MVTecDataset(
    root="data/mvtec",
    category="bottle",
    train=False
)

model = Autoencoder()

checkpoint = torch.load("checkpoints/epoch_9.pth")

model.load_state_dict(checkpoint["model"])

model.eval()


img = dataset[0].unsqueeze(0)

with torch.no_grad():

    recon = model(img)

score = reconstruction_error(img, recon)

print("Anomaly score:", score.item())