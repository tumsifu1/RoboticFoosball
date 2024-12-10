from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

class FoosballDataset(Dataset): 
    def __init__(self, images_dir, json_path, transform=None, train=True):

        self.train = train
        self.grid_size = 4
        self.images_dir = images_dir
        self.transform = transform
        with open (json_path, 'r') as f:
            self.data = json.load(f)

        self.preprocess = transforms.Compose([  # image is 1280x1280
            transforms.ToTensor(),           # Convert to PyTorch tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        #apply these to training dataset
        if train:
            self.augmentations = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.06, p=0.5),
                #A.HueSaturationValue(hue_shift_limit=1, sat_shift_limit=1, val_shift_limit=1, p=0.5)

            ])
        else:
            self.augmentations = None
    def __len__(self):
        return len(self.data)
    #2304 × 1296
    def __getitem__(self, idx):
        entry = self.data[idx]
        img_name = entry['image']
        ball_x, ball_y = entry['x'], entry['y']

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Apply preprocessing to all data
        image = self.preprocess(image)  # Always preprocess the image

        # Apply augmentations if training
        if self.augmentations and self.train:
            augmented = self.augmentations(image=image.permute(1, 2, 0).numpy())  # Convert to HWC for Albumentations
            image = torch.tensor(augmented['image']).permute(2, 0, 1).float()  # Convert back to CHW

        _, height, width = image.shape

        # Divide the image into regions
        region_height = height // self.grid_size
        region_width = width // self.grid_size

        # Determine the region containing the ball
        col_index = ball_x // region_width
        row_index = ball_y // region_height
        region_index = row_index * self.grid_size + col_index

        # Calculate the ball's relative coordinates within its region
        ball_x_rel = ball_x - col_index * region_width
        ball_y_rel = ball_y - row_index * region_height

        regions = []
        labels = []  # Store labels for each region
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                region = image[:, 
                            i * region_height:(i + 1) * region_height,
                            j * region_width:(j + 1) * region_width]
                regions.append(region)

                # Assign label: 1 if this is the ball region, else 0
                if i * self.grid_size + j == region_index:
                    labels.append(1)  # Ball region
                else:
                    labels.append(0)  # Non-ball region

        if self.train:
            # Select the region containing the ball
            labeled_region = regions[region_index]

            # Randomly select a region that does not contain the ball
            non_ball_indices = [i for i in range(len(regions)) if i != region_index]
            random_index = np.random.choice(non_ball_indices)
            random_region = regions[random_index]
            random_label = labels[random_index]

            # Return labeled and random regions with their labels
            return {
                "regions": torch.stack([labeled_region, random_region]),
                "labels": torch.tensor([1, random_label]),
                "relative_coords": (ball_x_rel, ball_y_rel)
            }

        else:
            # In test mode, return all regions and their labels
            return {
                "regions": torch.stack(regions),
                "labels": torch.tensor(labels)
            }