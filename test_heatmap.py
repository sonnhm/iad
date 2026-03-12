import torch

from datasets.mvtec import MVTecDataset
from models.autoencoder import Autoencoder
from visualization.heatmap import show_heatmap


dataset = MVTecDataset(
    root="data/mvtec",
    category="bottle",
    train=False
)

model = Autoencoder()

checkpoint = torch.load("checkpoints/epoch_9.pth")

model.load_state_dict(checkpoint["model"])

model.eval()


img = dataset[10].unsqueeze(0)

with torch.no_grad():

    recon = model(img)

show_heatmap(img, recon)