import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):

    def __init__(self, root, category, train=True, image_size=256):

        self.root = root
        self.category = category
        self.train = train

        self.image_paths = []
        self.labels = []

        if train:

            img_dir = os.path.join(root, category, "train", "good")

            for f in os.listdir(img_dir):

                if f.endswith(".png"):

                    self.image_paths.append(os.path.join(img_dir, f))
                    self.labels.append(0)

        else:

            test_dir = os.path.join(root, category, "test")

            for defect_type in os.listdir(test_dir):

                defect_dir = os.path.join(test_dir, defect_type)

                for f in os.listdir(defect_dir):

                    if f.endswith(".png"):

                        self.image_paths.append(
                            os.path.join(defect_dir, f)
                        )

                        if defect_type == "good":
                            self.labels.append(0)
                        else:
                            self.labels.append(1)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img = Image.open(self.image_paths[idx]).convert("RGB")

        img = self.transform(img)

        label = self.labels[idx]

        return img, label