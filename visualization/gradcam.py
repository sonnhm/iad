import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from datasets.mvtec import MVTecDataset
from models.cnn_feature import CNNFeatureExtractor


DATA_ROOT = "data/mvtec"
CATEGORY = "bottle"


dataset = MVTecDataset(
    root=DATA_ROOT,
    category=CATEGORY,
    train=False
)

img, label = dataset[10]

model = CNNFeatureExtractor()
model.eval()


img_batch = img.unsqueeze(0)

img_batch.requires_grad = True


features = model.features(img_batch)

output = features.mean()

model.zero_grad()

output.backward()


gradients = img_batch.grad[0].mean(dim=0).numpy()

heatmap = np.maximum(gradients, 0)

heatmap /= heatmap.max()


img_np = img.permute(1,2,0).numpy()

heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Input")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("GradCAM")
plt.imshow(heatmap, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(img_np)
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()