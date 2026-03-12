import torch

from models.autoencoder import Autoencoder


model = Autoencoder()

x = torch.randn(1,3,256,256)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)