from models.binaryClassifier.FoosballDataset import FoosballDataset
import torch


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
    def get_new_coordinates(self, x, y, row_index, col_index,region_width, region_height):
        new_x = x - (col_index*region_width)
        new_y = y - (row_index *region_height )

        return new_x, new_y
    
    def __get__item(self, idx):
        image, ball_exists, x, y = self.setupGetItem(idx)
        regions, region_height, region_width = self.breakImageIntoRegions(image)
        positive_region, region_index = self.getRegionWithBall(ball_exists, x, y, regions, region_height, region_width)
        new_x, new_y = self.get_new_coordinates(x, y, region_index, region_index, region_width, region_height)
        
        return {
            "region": positive_region,
            "x": new_x,
            "y": new_y
        }
    

    