from datasets.mvtec import MVTecDataset
from models.autoencoder import Autoencoder
from training.trainer import Trainer


dataset = MVTecDataset(
    root="data/mvtec",
    category="bottle",
    train=True
)

model = Autoencoder()

trainer = Trainer(
    model=model,
    dataset=dataset,
    batch_size=32,
    epochs=10
)

trainer.train()