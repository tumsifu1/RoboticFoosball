import numpy as np

def rebuild_absolute_coordinates(region_row, region_col, local_x, local_y, region_width, region_height):
    absolute_x = region_col * region_width + local_x
    absolute_y = region_row * region_height + local_y
    return absolute_x, absolute_y

import torch
import numpy as np

def segment_image(image: torch.Tensor, grid_size=8):
    assert image.dim() == 3, "Input image must be a 3D tensor of shape (C, H, W)"

    C, H, W = image.shape  # Get channels, height, and width
    region_width = W // grid_size
    region_height = H // grid_size

    regions = []
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate slicing coordinates
            start_y = row * region_height
            end_y = (row + 1) * region_height
            start_x = col * region_width
            end_x = (col + 1) * region_width

            # Extract the region using PyTorch slicing
            region = image[:, start_y:end_y, start_x:end_x]  # Shape: (C, region_H, region_W)
            regions.append(region)

    return regions
