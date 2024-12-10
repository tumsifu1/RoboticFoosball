from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
import torch
import cv2
import numpy as np

class FoosballDataset(Dataset): 
    def __init__(self, images_dir, json_path, transform=None, train=True):

        self.train = train

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
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
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

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        #print(image.size)
        #apply preprocessing

        image = self.preprocess(image)
        _, height, width = image.shape
        print(image.shape) #(3, 1296, 2304)
        if isinstance(image, torch.Tensor):
            image = image.permute(0, 1, 2).numpy() # (H, W, C)
        #print(image.shape)
       

        if self.train and self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
            
        
        image = torch.tensor(image).permute(2, 0, 1).float() #convert to tensor and change the shape to (C, H, W)
        #get the height and width of the image
        _, height, width = image.shape
        
        return image, height, width
            