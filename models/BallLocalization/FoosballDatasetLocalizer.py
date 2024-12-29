from models.binaryClassifier.FoosballDataset import FoosballDataset

import torch
import matplotlib.pyplot as plt

class FoosballDatasetLocalizer(FoosballDataset):

    def __init__(self, images_dir, json_path, transform=None, train=True):
        super().__init__(images_dir, json_path, transform, train)
    def collate_fn(batch):
        # Stack the 'region' tensors into a single batch
        regions = torch.stack([item['region'] for item in batch], dim=0)  # Assumes 'region' is a tensor

        # Collect 'x' and 'y' into separate tensors
        x_coords = torch.tensor([item['x'] for item in batch], dtype=torch.float32)
        y_coords = torch.tensor([item['y'] for item in batch], dtype=torch.float32)

        return regions, x_coords, y_coords
    def get_new_coordinates(self, x, y,region_width, region_height):
        print("Get new coordinates")
        col_index = x // region_width
        row_index = y // region_height
        print(f"X: {x}, Y: {y}")
        print(f"Row Index: {row_index}, Col Index: {col_index}")
        new_x = x - (col_index*region_width)
        new_y = y - (row_index *region_height)

        if new_x < 0 or new_y < 0:
            raise ValueError(f"Invalid coordinates: ({new_x}, {new_y})")

        return new_x, new_y


    def __getitem__(self, idx):

        image, ball_exists, x, y = self.setupGetItem(idx)
        #plt.imshow(image)
        #plt.scatter(x, y, c='r', s=5)
        #plt.show()
        image = self.preProcessImage(image)

        regions,region_width, region_height = self.breakImageIntoRegions(image)
        positive_region, region_index = self.getRegionWithBall(ball_exists, x, y, regions, region_height, region_width)
        new_x, new_y = self.get_new_coordinates(x, y, region_width, region_height)
        return positive_region, new_x, new_y, ball_exists

    

    