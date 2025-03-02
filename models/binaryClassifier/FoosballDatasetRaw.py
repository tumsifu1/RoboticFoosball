from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import torch

class FoosballDatasetRaw(Dataset): 
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load JSON data
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.data = [{"image": key, **value} for key, value in raw_data.items()] if isinstance(raw_data, dict) else raw_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        img_name = str(entry['image'])  # Ensure it's a string
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert('RGB')  # Open image
        if self.transform:
            image = self.transform(image)  # Apply transform (ToTensor only)

        return image 