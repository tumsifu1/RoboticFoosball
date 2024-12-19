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
import random

class FoosballDataset(Dataset): 
    def __init__(self, images_dir, json_path, transform=None, train=True):

        self.train = train
        self.grid_size = 4
        self.images_dir = images_dir
        self.transform = transform
        with open (json_path, 'r') as f:
            self.data = json.load(f)

        self.preprocess = transforms.Compose([  # image is 1280x1280
            #transforms.ToTensor(),           # Convert to PyTorch tensor (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        #apply these to training dataset
        if train:
            self.augmentations = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # Subtle brightness/contrast tweaks
                A.GaussNoise(std_range=(0.0,0.1),mean_range=(0,0), p=0.2),  # Reduced noise intensity and probability
                #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0.5, p=0.2),  # Gentle color adjustments
                A.Blur(blur_limit=3, p=0.1)  # Very occasional slight blur
            ])
        else:
            self.augmentations = None
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.data)
    
    def collate_fn( batch):
        #flatten paired regions and labels

        regions = torch.cat([item['regions'] for item in batch], dim=0)
        labels = torch.cat([item['labels'] for item in batch], dim=0)

        #shuffle regions and labels
        combined = list(zip(regions, labels))
        random.shuffle(combined)
        regions, labels = zip(*combined)

        return torch.stack(regions),torch.tensor(labels)
    
    
    def collate_fn( batch):
        #flatten paired regions and labels

        regions = torch.cat([item['regions'] for item in batch], dim=0)
        labels = torch.cat([item['labels'] for item in batch], dim=0)

        #shuffle regions and labels
        combined = list(zip(regions, labels))
        random.shuffle(combined)
        regions, labels = zip(*combined)

        return torch.stack(regions),torch.tensor(labels)
    
    #2304 × 1296
    def __getitem__(self, idx):
        entry = self.data[idx]
        img_name = entry['image']
        ball_exists = entry['ball_exists']
        ball_x = None
        ball_y = None

        ball_exists = entry['ball_exists']
        ball_x = None
        ball_y = None


        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        #print(f"Image: {img_name}, Ball Exists: {ball_exists}")
        # Apply preprocessing to all data
        
        #conmvert to tensor
        image = self.to_tensor(image)
        # Apply augmentations if training
        if self.augmentations and self.train:
            augmented = self.augmentations(image=image.permute(1, 2, 0).numpy())  # Convert to HWC for Albumentations
            image = torch.tensor(augmented['image']).permute(2, 0, 1).float()  # Convert back to CHW
        image = self.preprocess(image)  # Always preprocess the image
        
        _, height, width = image.shape
                # Divide the image into regions
                # Divide the image into regions
        region_height = height // self.grid_size
        region_width = width // self.grid_size

        # Divide the image into regions
        # Divide the image into regions
        regions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                region = image[:, 
                            i * region_height:(i + 1) * region_height,
                            j * region_width:(j + 1) * region_width]
                regions.append(region)
        # Find positive (ball) region
        if ball_exists:
            ball_x, ball_y = entry['x'], entry['y']
            #print(f"Region Width: {region_width}, Region Height: {region_height}")
            col_index = ball_x // region_width
            row_index = ball_y // region_height
            #print(f"Row: {row_index}, Col: {col_index}")
            region_index = row_index * self.grid_size + col_index
            if region_index < 0 or region_index >= len(regions):
                raise IndexError(f"Invalid region_index: {region_index}. Valid range is 0 to {len(regions)-1}.")

            
            positive_region = regions[region_index]
        else:
            positive_region = None  # No ball in this image

        # Select a random negative (no ball) region
        if ball_exists:
            negative_indices = [i for i in range(len(regions)) if i != region_index]
        else:
            negative_indices = list(range(len(regions)))
        random_negative_index = np.random.choice(negative_indices)
        negative_region = regions[random_negative_index]

        # Prepare labels
        positive_label = 1 if ball_exists else 0
        negative_label = 0

        # Return positive and negative samples
        #print(f"Positive Region: {positive_region}, Negative Region: {negative_region}, Positive Label: {positive_label}, Negative Label: {negative_label}")
        return {
            "regions": torch.stack([positive_region, negative_region]),
            "labels": torch.tensor([positive_label, negative_label]),
        }