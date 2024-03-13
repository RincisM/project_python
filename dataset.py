import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class WareHouseDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # Map each unique value in mask to a unique number in the range 0-1
        unique_mask_values = np.unique(mask)
        num_unique_values = len(unique_mask_values)
        self.unique_values = {value: i / (num_unique_values - 1) for i, value in enumerate(unique_mask_values)}

        # Assign unique numbers to mask values
        for value, unique_number in self.unique_values.items():
            mask[mask == value] = unique_number

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask