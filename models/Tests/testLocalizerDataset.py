import sys
import os

from models.BallLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

json_path = "data/labels/labels.json"
images_dir = "data/images"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = FoosballDatasetLocalizer(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle= False, collate_fn=FoosballDatasetLocalizer.collate_fn)
#2304 × 1296

def unnormalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    """Undo normalization for visualization."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean #tensor-wise math 

def test_get_new_coordinates():
    # Test parameters
    x= 1098
    y = 652
    row_width = 576  # 2304 / 4
    row_height = 324  # 1296 / 4
    row_index = y // row_height  # Compute row index
    col_index = x // row_width   # Compute column index

    # Expected new coordinates
    expected_new_x = x - (col_index * row_width)
    expected_new_y = y - (row_index * row_height)

    # Call the method

    new_x, new_y = dataset.get_new_coordinates(x, y, row_index, col_index, row_width, row_height)

    # Validate the result
    assert new_x == expected_new_x, f"Expected new_x={expected_new_x}, but got {new_x}"
    assert new_y == expected_new_y, f"Expected new_y={expected_new_y}, but got {new_y}"

    # Visualization (optional, for manual inspection)
    for batch in dataloader:
        print(batch["region"].shape)
        break
        # Get a single image and label
        image = batch['region']
        x, y = batch['x'], batch['y']
        print(f"Original Coordinates: {x}, {y}")
        print(f"New Coordinates: {new_x}, {new_y}")

        # Unnormalize the image
        unnormalized_img = unnormalize(image.unsqueeze(0))  # Add batch dim
        img = unnormalized_img.squeeze().permute(1, 2, 0).numpy()  # Convert to HWC

        # Convert to BGR format for OpenCV 
        img_bgr = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

        # Display the image
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        #plt.scatter(x, y, c='red', label='Original')
        plt.scatter(new_x, new_y, c='blue', label='New')
        plt.legend()
        plt.axis("off")
        plt.title(f"Test: Original ({x},{y}) -> New ({new_x},{new_y})")
        plt.show()
    
    

def __main__():
    test_get_new_coordinates()   

if __name__=="__main__":
    __main__()