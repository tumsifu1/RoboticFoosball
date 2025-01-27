import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from src.tools.labelling_tool import numerical_sort
from src.tools.unnormalize import unnormalize
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer

# Paths
json_path = "data/train/labels/labels.json"
images_dir = "data/train/images"

labeled_data = []
img_index = [-1]  # Mutable index for the current image

# Load and sort image files
image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
image_files.sort(key=numerical_sort)
#TODO: add image labels
#TODO: add old x and y and new x and y
#TODO: tolerance calculation
#TODO: find new infrence speed
# Process each image
for img_name in image_files:
    dataset = FoosballDatasetLocalizer(json_path=json_path, images_dir=images_dir, transform=None)
    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert('RGB')
    img_index[0] += 1

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    ballData = data[img_name]

    x, y = ballData["x"], ballData["y"]
    labeled_data.append((image, x, y))

    new_x, new_y = dataset.get_new_coordinates(x, y, 224, 224)

    img_arr = np.array(image)
    pre_image = dataset.preprocessImage(image)

    # Break image into regions and find region containing ball
    regions, region_width, region_height = dataset.breakImageIntoRegions(pre_image)
    _, region_index = dataset.getRegionWithBall(1, x, y, regions, region_height, region_width)

    # Grid display parameters
    ROWS, COLS = 4, 4
    fig, ax = plt.subplots(ROWS, COLS, figsize=(8, 8)) 
    

    for row in range(ROWS):
        for col in range(COLS):
            x_start = col * (img_arr.shape[1] // COLS)
            x_end = (col + 1) * (img_arr.shape[1] // COLS)
            y_start = row * (img_arr.shape[0] // ROWS)
            y_end = (row + 1) * (img_arr.shape[0] // ROWS)

            if row * COLS + col == region_index:
                # Use the corresponding image from the regions array
                segment = img_arr[y_start:y_end, x_start:x_end]
                

                # Display the updated region with contours
                ax[row, col].imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))

                # Scatter plot on the image
                ax[row, col].scatter(new_x, new_y, c='r', s=10)
                ax[row, col].title.set_text(f"Region {row * COLS + col}")
            else:
                # Slice and show the original segment
                ax[row, col].title.set_text(f"Region {row * COLS + col}")
                segment = img_arr[y_start:y_end, x_start:x_end]
                ax[row, col].imshow(segment)

            # Remove axis ticks for cleaner visualization
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

    # Adjust layout to remove gaps
   
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()