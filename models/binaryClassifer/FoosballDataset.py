from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import torch

class FoosballDataset(Dataset): 
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open (json_path, 'r') as f:
            self.data = json.load(f)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        img_name = entry['image']
        label = entry['ball_exists']

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)