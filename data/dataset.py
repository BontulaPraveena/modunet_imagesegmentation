import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        all_images = os.listdir(image_dir)
        
        # Ensure only images with a corresponding mask are added to the valid list
        self.valid_images = [img for img in all_images if os.path.exists(os.path.join(mask_dir, img.replace('.jpg', '_mask.jpg')))]
        
        if len(self.valid_images) == 0:
            raise RuntimeError("No valid image-mask pairs were found. Check your dataset directories and naming conventions.")
        
    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.valid_images[idx])
        mask_path = os.path.join(self.mask_dir, self.valid_images[idx].replace('.jpg', '_mask.jpg'))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
