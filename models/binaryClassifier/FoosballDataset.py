import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
import os
import json
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
class FoosballDataset(Dataset): 
    GRID_SIZE = 8 # 4x4 grid
    REGION_WIDTH = 2304//GRID_SIZE
    REGION_HEIGHT = 1296//GRID_SIZE
    WIDTH = 2304
    HEIGHT =1296
    @classmethod
    def get_region_height(cls):
        return cls.REGION_HEIGHT
    @classmethod
    def get_region_width(cls):
        return cls.REGION_WIDTH
    @classmethod
    def get_grid_size(cls):
        return cls.GRID_SIZE
    
    def __init__(self, images_dir, json_path, transform=None, train=True):
        COMPUTED_MEAN = [0.1249, 0.1399, 0.1198]
        COMPUTED_STD = [0.1205, 0.1251, 0.1123]
        self.train = train

        self.images_dir = images_dir
        self.transform = transform

        self.data = None 
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            # Transform dictionary to a list if necessary
            if isinstance(raw_data, dict):
                self.data = [{"image": key, **value} for key, value in raw_data.items()]
            else:
                self.data = raw_data

        self.preprocess = transforms.Compose([
            #transforms.ToTensor(),           # Convert to PyTorch tensor (C, H, W) comment out when testing
            transforms.Normalize(mean=COMPUTED_MEAN, std=COMPUTED_STD)  # Normalize
        ])
        #pixel augmentations for training classifer model

        self.pixelAugmentations = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.1), contrast_limit=(-0.2,0.2), p=0.3),  # Subtle brightness/contrast tweaks
            A.GaussNoise(std_range=(0.0,0.5),mean_range=(0,0), p=0.2),  # Reduced noise intensity and probability
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0.5, p=0.2),  # Gentle color adjustments
            A.Blur(blur_limit=(3,7), p=0.5)  # Very occasional slight blur
        ])

        #spatial augmentations for training localizer model
        self.spatialAugmentations = A.Compose(
            [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit= 0.02, scale_limit=0.05, rotate_limit=0, p=0.5),
        ],
            keypoint_params=A.KeypointParams(format='xy') # Ensure x and y coordinates are transformed 
        )

        self.aug_pipeline = A.Compose(
            self.pixelAugmentations.transforms + self.spatialAugmentations.transforms,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)
    
    def collate_fn( batch):
        """turns data for getitem to tuple of tensors"""
        #flatten paired regions and labels

        regions = torch.cat([item['regions'] for item in batch], dim=0)
        labels = torch.cat([item['labels'] for item in batch], dim=0)

        #shuffle regions and labels
        combined = list(zip(regions, labels))
        random.shuffle(combined)
        regions, labels = zip(*combined)


        return torch.stack(regions),torch.tensor(labels)
    
    def getRegionWithBall(self, ball_exists, x, y, regions):
        """returns the region containing the ball and the region index"""
        # These calculations are correct for a 4x4 grid and fixed dimensions
        # self.REGION_HEIGHT = 1296 // 4 = 324
        # self.REGION_WIDTH = 2304 // 4 = 576

        #for checking the total image dimensions
        #total_width = self.REGION_WIDTH * GRID_SIZE_SIZE
        #total_height = self.REGION_HEIGHT * (len(regions) // GRID_SIZE_SIZE)
        #print(f"Total Image Dimensions: width={total_width}, height={total_height}")

        if ball_exists:
            ball_x, ball_y = x, y

            # Calculate col_index (this is correct)
            col_index = ball_x // self.REGION_WIDTH
            #print(f"Col Index: {col_index} ball_x: {ball_x} self.REGION_WIDTH: {self.REGION_WIDTH}")

            # Calculate row_index (this is correct)
            row_index = ball_y // self.REGION_HEIGHT
            #print(f"Row Index: {row_index} ball_y: {ball_y} self.REGION_HEIGHT: {self.REGION_HEIGHT}")
            # Ensure row_index and col_index are within bounds
            row_index = min(row_index, self.GRID_SIZE - 1)
            col_index = min(col_index, self.GRID_SIZE - 1)

            # Calculate region_index
            region_index = int(row_index * self.GRID_SIZE + col_index)
            #print(f"self.GRID_SIZE: {self.GRID_SIZE}, Region Index: {region_index}")

            if region_index < 0 or region_index >= len(regions):
                raise IndexError(f"Invalid region_index: {region_index}. Valid range is 0 to {len(regions)-1}.")

            positive_region = regions[region_index]

        else:
            raise ValueError("Ball does not exist in the image.")

        return positive_region, region_index


    def getRandomNegativeRegion(self,ball_exists, pos_region_index, regions):
        """returns a random negative region for classification"""
            # Select a random negative (no ball) region
        if ball_exists:
            negative_indices = [i for i in range(len(regions)) if i != pos_region_index]
        else:
            negative_indices = list(range(len(regions)))
            raise ValueError("Ball does not exist in the image.")
        
        random_negative_index = np.random.choice(negative_indices)
        negative_region = regions[random_negative_index]
        return negative_region
    
    def breakImageIntoRegions(self, image):
        """Break the image into regions and return a list of regions"""
        # _, height, width = image.shape
        # assert height % self.GRID_SIZE == 0, "Image height is not divisible by grid_size"
        # assert width % self.GRID_SIZE == 0, "Image width is not divisible by grid_size"

        # Divide the image into regions
        # Divide the image into regions
        regions = []
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                region = image[:, 
                            i * self.REGION_HEIGHT:(i + 1) * self.REGION_HEIGHT, # take all channels, take ith region height to i+1th region height
                            j * self.REGION_WIDTH:(j + 1) * self.REGION_WIDTH]
                regions.append(region)
                #print(f"Row {i}, Col {j} -> Index {len(regions) - 1}")
        return regions
    
    
    def setupGetItem(self, idx):
        """Setup the __getitem__ method by loading the image and extracting the ball coordinates"""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.data)}")
        #print(type(self.data))
        transform = transforms.ToTensor()
        entry = self.data[idx]
        img_name = entry['image']
        img_name = str(img_name)

        ball_exists = entry['ball_exists']
        x, y = entry['x'],  entry['y']

        img_path = os.path.join(self.images_dir, img_name)
        image = transform(Image.open(img_path).convert('RGB'))
        #print(f"Image: {img_name}, Ball Exists: {ball_exists}")

        if x == 1:
            raise ValueError(f"Invalid coordinates at setUpgetItem: ({x}, {y})")
        return image, ball_exists, x, y, img_name
    
    def preprocessImage(self, image,keypoints):
        """Preprocess the image by applying transformations and augmentations"""
        #conmvert to tensor
        # Apply augmentations if training
        x, y = keypoints[0], keypoints[1]
        if self.train and False:
            keypoints = [keypoints] #wrap in list
            augmented = self.aug_pipeline(image=image.permute(1, 2, 0).numpy(),keypoints=keypoints)
            image = torch.tensor(augmented['image']).permute(2, 0, 1).float()  # Convert back to CHW
            if not augmented:
                print("augmented is empty")
            augmented_keypoints = augmented["keypoints"][0]
            x,y = augmented_keypoints[0], augmented_keypoints[1]

        image = self.preprocess(image)  # Always preprocess the image

        return image,x,y
    
    def returnLabels(self, ball_exists, positive_region, negative_region):
        """for return the regions as a stack of tensors and the labels as a tensor"""
        positive_label = 1 if ball_exists else 0
        negative_label = 0

        # Return positive and negative samples
        return {
            "regions": torch.stack([positive_region, negative_region]),
            "labels": torch.tensor([positive_label, negative_label]),
        }
    
    #2304 × 1296
    def __getitem__(self, idx):

        image, ball_exists, x, y, _= self.setupGetItem(idx)
        # Apply preprocessing to all data
        
        image, x,y = self.preprocessImage(image, (x,y))
        
        regions = self.breakImageIntoRegions(image)
        # Find positive (ball) region

        positive_region, pos_region_index = self.getRegionWithBall(ball_exists, x, y, regions)
        # Select a random negative (no ball) region

        negative_region = self.getRandomNegativeRegion(ball_exists, pos_region_index, regions)

        # Return positive and negative samples
        return self.returnLabels(ball_exists, positive_region, negative_region)
        #print(f"Positive Region: {positive_region}, Negative Region: {negative_region}, Positive Label: {positive_label}, Negative Label: {negative_label}")
        