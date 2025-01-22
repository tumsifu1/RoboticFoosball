import sys
import os
from PIL import Image
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer
from src.tools.unnormalize import unnormalize
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import json
json_path = "data/train/labels/labels.json"
images_dir = "data/train/images/"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = FoosballDatasetLocalizer(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle= True, collate_fn=FoosballDatasetLocalizer.collate_fn)
#2304 × 1296

def test_getRegionWithBall_realData(): #You must comment out to tensor in the preprocess
    """Test the get_region_with_ball method with real data."""
    img_name = "img_925.jpg"
    img_path = "data/images/" + img_name
    image = Image.open(img_path).convert('RGB')
    image = dataset.preprocessImage(image)
    print(f"Image Shape After Preprocessing: {image.shape}")
    ball_exists = True
    #get x and why in the json at img_0.jpg
    x = 0
    y = 0
    with open(json_path, 'r') as f:
        data = json.load(f)

    ballData = data[img_name]

    x = ballData["x"]
    y = ballData["y"]

    regions, region_width, region_height = dataset.breakImageIntoRegions(image)
    positive_region, region_index = dataset.getRegionWithBall(ball_exists, x, y, regions, region_height, region_width)
    new_x, new_y = dataset.get_new_coordinates(x, y, region_width, region_height)
    positive_region = unnormalize(positive_region)

    plt.imshow(positive_region.permute(1, 2, 0))

    plt.axis("off")
    plt.title(f"Ball Region and index: {region_index}")
    plt.xlabel(f"Original ({x},{y}) -> New ({new_x},{new_y})")
    plt.scatter(new_x, new_y, c='r', s=50, label='Ball')
    #remove pre process and normilization)
    plt.show()
            
def test_get_new_coordinates():#You must comment out to tensor in the preprocess
    """Test the get_new_coordinates method without the full dataloader"""
    img_name = "img_4033.jpg"
    img_path = images_dir + img_name
    image = Image.open(img_path).convert('RGB')
    image = dataset.preprocessImage(image)
    ball_exists = True 
    x = 0
    y = 0
    with open(json_path, 'r') as f:
        data = json.load(f)  # Load JSON into 'data'

    ballData = data[img_name]  # Get the ball data for the image
    x = ballData["x"]  # Get the x-coordinate of the ball
    y = ballData["y"]  # Get the y-coordinate of the ball

    region_width = 576  # 2304 / 4
    region_height = 324  # 1296 / 4
    row_index = y // region_height  # Compute row index
    col_index = x // region_width   # Compute column index
    print("test_get_new_coordinates")
    print(f"Row Index: {row_index}, Col Index: {col_index}")
    print(f"Region Width: {region_width}, Region Height: {region_height}")
    
    # Expected new coordinates
    expected_new_x = x - (col_index * region_width)
    expected_new_y = y - (row_index * region_height)
    print(f"Expected new_x: {expected_new_x}, Expected new_y: {expected_new_y}")
    new_x, new_y = dataset.get_new_coordinates(x, y, region_width, region_height)

    # Validate the result
    assert new_x == expected_new_x, f"Expected new_x={expected_new_x}, but got {new_x}"
    assert new_y == expected_new_y, f"Expected new_y={expected_new_y}, but got {new_y}"
    
    regions,region_width, region_height = dataset.breakImageIntoRegions(image)
    positive_region, _ = dataset.getRegionWithBall(ball_exists, x, y, regions, region_height, region_width)
    unnormalized_img = unnormalize(positive_region)

    #plt.scatter(x, y, c='red', label='Original')
    plt.scatter(new_x, new_y, c='blue', label='New')
    plt.imshow(unnormalized_img.permute(1, 2, 0))
    plt.legend()
    plt.axis("off")
    plt.title(f"Test: Original ({x},{y}) -> New ({new_x},{new_y})")
    plt.show()

#loop through the dataloader and display the segments with ball see where the ball is
def full_test():
    for image, x,y, img_name in dataloader:
        
        # Unnormalize the image
        unnormalized_img = unnormalize(image.unsqueeze(0))  # Add batch dim
        img = unnormalized_img.squeeze().permute(1, 2, 0).numpy()  # Convert to HWC
        print(f"x: {x}, y: {y}")
        img_bgr = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        
        # Display the image
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        #plt.scatter(x, y, c='red', label='Original')
        plt.scatter(x, y, c='blue', label='New')
        plt.title(f"Image: {img_name}")
        plt.axis("off")
        plt.show()

def __main__():
    """Run full test to see the issue with dataset"""
    test_getRegionWithBall_realData()
    #test_get_new_coordinates()  
    #full_test() 

if __name__=="__main__":
    __main__()