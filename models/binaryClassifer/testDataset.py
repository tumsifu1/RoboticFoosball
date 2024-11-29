import os
from torch.utils.data import DataLoader
import torch
import json
from torchvision import transforms
from FoosballDataset import FoosballDataset

json_path = "data/labeled_data_part2.json"
images_dir = "data/labeled_images_part2"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Test the DataLoader
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Images shape: {images.shape}")  # Should be [batch_size, 3, H, W]
    print(f"Labels: {labels}")  # Should be tensor of [batch_size]
