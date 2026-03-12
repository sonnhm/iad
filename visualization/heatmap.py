import torch
import matplotlib.pyplot as plt

from datasets.mvtec import MVTecDataset
from models.autoencoder import Autoencoder


DATA_ROOT = "data/mvtec"
CATEGORY = "bottle"


dataset = MVTecDataset(
    root=DATA_ROOT,
    category=CATEGORY,
    train=False
)

img, label = dataset[10]

model = Autoencoder()

checkpoint = torch.load("checkpoints/epoch_9.pth", map_location="cpu")

model.load_state_dict(checkpoint["model"])

model.eval()


img_batch = img.unsqueeze(0)

with torch.no_grad():
    recon = model(img_batch)


error = (img_batch - recon) ** 2

heatmap = error.mean(dim=1).squeeze().numpy()


plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(img.permute(1,2,0))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Reconstruction")
plt.imshow(recon.squeeze().permute(1,2,0))
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Anomaly Heatmap")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.show()