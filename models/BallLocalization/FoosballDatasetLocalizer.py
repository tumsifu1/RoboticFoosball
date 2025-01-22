
from models.binaryClassifier.FoosballDataset import FoosballDataset

import torch
import matplotlib.pyplot as plt
import numpy
class FoosballDatasetLocalizer(FoosballDataset):

    def __init__(self, images_dir, json_path, transform=None, train=True):
        super().__init__(images_dir, json_path, transform, train)
    def collate_fn(batch):
        # Unpack the batch
        positive_regions = torch.stack([item[0] for item in batch], dim=0)  # Stack all positive_region tensors
        x_coords = torch.tensor([item[1] for item in batch], dtype=torch.float32)  # Collect all new_x values
        y_coords = torch.tensor([item[2] for item in batch], dtype=torch.float32)  # Collect all new_y values
        img_names = [item[4] for item in batch]  # Collect all image names
        #ball_exists = torch.tensor([item[3] for item in batch], dtype=torch.float32)  # Collect all ball_exists values

        return positive_regions, x_coords, y_coords, img_names
    def preprocessImage(self, image):
        """Preprocess the image by applying transformations and augmentations"""
        image = numpy.array(image)
        # Apply augmentations if training
        if self.pixelAugmentations and self.spatialAugmentations and self.train:
            image = self.pixelAugmentations(image=image)['image']  # Apply pixel-level augmentations
            image = self.spatialAugmentations(image=image)['image']  # Apply spatial augmentations
        
        #conmvert to tensor
        image = self.to_tensor(image).float()

        image = self.preprocess(image)  # Always preprocess the image
        return image
    
    def get_new_coordinates(self, x, y,region_width, region_height):
        #print("Get new coordinates")
        col_index = x // region_width
        row_index = y // region_height
        #print(f"X: {x}, Y: {y}")
        #print(f"Row Index: {row_index}, Col Index: {col_index}")
        new_x = (x - (col_index*region_width))
        new_y = (y - (row_index *region_height))
        if new_x < 0 or new_y < 0:
            raise ValueError(f"Invalid coordinates: ({new_x}, {new_y})")

        return new_x, new_y


    def __getitem__(self, idx):

        image, ball_exists, x, y, img_name = self.setupGetItem(idx)
        #plt.imshow(image)
        #plt.scatter(x, y, c='r', s=5)
        #plt.show()
        image = self.preprocessImage(image)

        regions,region_width, region_height = self.brsaaeakImageIntoRegions(image)
        positive_region, region_index = self.getRegionWithBall(ball_exists, x, y, regions, region_height, region_width)
        new_x, new_y = self.get_new_coordinates(x, y, region_width, region_height)
        return positive_region, new_x, new_y, ball_exists, img_name

    

    