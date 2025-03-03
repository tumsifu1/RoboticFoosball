import numpy as np

def rebuild_absolute_coordinates(region_row, region_col, local_x, local_y, region_width, region_height):
    absolute_x = region_col * region_width + local_x
    absolute_y = region_row * region_height + local_y
    return absolute_x, absolute_y

def segment_image(image: np.ndarray):
    
    grid_size = 4
    region_width = 1280//grid_size
    region_height = 720//grid_size

    regions = []
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate the slicing coordinates
            start_y = row * region_height
            end_y = (row + 1) * region_height
            start_x = col * region_width
            end_x = (col + 1) * region_width
            region = image[start_y:end_y, start_x:end_x]
            regions.append(region)
    return regions