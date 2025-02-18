
from models.binaryClassifier.FoosballDataset import FoosballDataset

import torch
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F
class FoosballDatasetLocalizer(FoosballDataset):

    def __init__(self, images_dir, json_path, transform=None, train=True):
        super().__init__(images_dir, json_path, transform, train)
    def collate_fn(batch):
        # Unpack the batch
        positive_regions = torch.stack([item[0] for item in batch], dim=0)  # Stack all positive_region tensors
        x_coords = torch.stack([torch.tensor(item[1], dtype=torch.float32) for item in batch])
        y_coords = torch.stack([torch.tensor(item[2], dtype=torch.float32) for item in batch])
        old_x_coord = torch.stack([torch.tensor(item[3], dtype=torch.float32) for item in batch])
        old_y_coord = torch.stack([torch.tensor(item[4], dtype=torch.float32) for item in batch])

        
        img_names = [item[5] for item in batch]  # Collect all image names
        #ball_exists = torch.tensor([item[3] for item in batch], dtype=torch.float32)  # Collect all ball_exists values

        return positive_regions, x_coords, y_coords, old_x_coord, old_y_coord, img_names
    
    def get_new_coordinates(self, x, y):
        #print("Get new coordinates")
        col_index = x // self.get_region_width()
        row_index = y // self.get_region_height()
        #print(f"X: {x}, Y: {y}")
        #print(f"Row Index: {row_index}, Col Index: {col_index}")
        new_x = (x - (col_index*self.get_region_width()))
        new_y = (y - (row_index*self.get_region_height()))
        if new_x < 0 or new_y < 0:
            raise ValueError(f"Invalid coordinates: ({new_x}, {new_y})")

        return new_x, new_y

    def rescale_image(self, positive_region):
            positive_region_resized = F.interpolate(positive_region.unsqueeze(0), 
                                        size=(227, 227), 
                                        mode='bilinear', 
                                        align_corners=False).squeeze(0)

            return positive_region_resized
    
    def rescale_coordinates(self, new_x, new_y):
        
        scaled_x = (new_x / self.get_region_width()) *227
        #print(f"scaled_after 227 x: {scaled_x}")
        scaled_y = (new_y / self.get_region_height()) *227
        return scaled_x, scaled_y

    def __getitem__(self, idx):

        image, ball_exists, x, y, img_name = self.setupGetItem(idx)
        #plt.imshow(image)
        #plt.scatter(x, y, c='r', s=5)
        #plt.show()
        
        #print(f"Original X, Y: ({x}, {y})")
        regions = self.breakImageIntoRegions(image)

        positive_region, _ = self.getRegionWithBall(ball_exists, x, y, regions)
        new_x, new_y = self.get_new_coordinates(x, y)
        #print(f"New X, Y (Before Scaling): ({new_x}, {new_y})")
        image, x,y = self.preprocessImage(positive_region, (new_x,new_y))
        positive_region_scaled = self.rescale_image(positive_region)
        
        

        new_x_scaled, new_y_scaled = self.rescale_coordinates(new_x, new_y)
        #print(f"New X, Y (After Scaling): ({new_x_scaled}, {new_y_scaled})")

        return positive_region_scaled, new_x_scaled, new_y_scaled, new_x, new_y, img_name

    

    