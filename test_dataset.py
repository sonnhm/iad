import os
print(os.getcwd())
from datasets.mvtec import MVTecDataset

dataset = MVTecDataset(
    root="data/mvtec",
    category="bottle",
    train=True
)

print(len(dataset))

img = dataset[0]

print(img.shape)