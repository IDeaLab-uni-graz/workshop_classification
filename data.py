import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(image_filename)[0] + ".txt"
        )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load label (NumPy array)
        label = np.genfromtxt(label_path, delimiter=' ')

        # Apply transforms
        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label

    def get_all_labels(self):
        labels = []
        for filename in self.image_filenames:
            label_path = os.path.join(
                self.label_dir,
                os.path.splitext(filename)[0] + ".txt"
            )
            label = np.genfromtxt(label_path, delimiter=' ')

            if self.label_transform:
                label = self.label_transform(label)

            labels.append(label)
        return np.array(labels)