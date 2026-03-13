import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from datasets.mvtec import MVTecDataset
from models.patchcore import PatchCore


DATA_ROOT = "data/mvtec"
CATEGORY = "bottle"


train_dataset = MVTecDataset(
    root=DATA_ROOT,
    category=CATEGORY,
    train=True
)

test_dataset = MVTecDataset(
    root=DATA_ROOT,
    category=CATEGORY,
    train=False
)


model = PatchCore()
model.eval()


memory_bank = []

for img, _ in train_dataset:

    img = img.unsqueeze(0)

    with torch.no_grad():
        patches = model(img)

    memory_bank.append(patches.numpy())

memory_bank = np.concatenate(memory_bank, axis=0)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(memory_bank)


img, label = test_dataset[10]

img_batch = img.unsqueeze(0)

with torch.no_grad():
    patches = model(img_batch)

patches = patches.numpy()

dist, _ = nn.kneighbors(patches)

size = int(np.sqrt(len(dist)))
heatmap = dist.reshape(size, size)   # patch grid

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(img.permute(1,2,0))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Patch Anomaly Score")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(img.permute(1,2,0))
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()