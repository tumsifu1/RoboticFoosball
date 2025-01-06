import os
from torch.utils.data import DataLoader
import torch
import json
from torchvision import transforms
from models.binaryClassifier.FoosballDataset import FoosballDataset
from src.tools.unnormalize import unnormalize
import matplotlib.pyplot as plt
import cv2
from PIL import Image
json_path = "data/labels/labels.json"
images_dir = "data/images"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle= False, collate_fn=FoosballDataset.collate_fn)
print(f"Dataset size: {len(dataset)}")

def visulize_image():
    """Visualize a full image from the dataset."""
    img_name = "img_0.jpg"
    img_path = "data/images/" + img_name
    image = Image.open(img_path).convert('RGB')
    image = dataset.preProcessImage(image)
    print(f"Image Shape After Preprocessing: {image.shape}")
    ball_exists = True
    #get x and why in the json at img_0.jpg
    x = 0
    y = 0
    with open(json_path, 'r') as f:
        data = json.load(f)  # Load JSON into 'data'

    # 'data' is a list
    for item in data:
        if item.get("image") == img_name:  # Find the entry for "img_0.jpg"
            x = item.get("x")
            y = item.get("y")
            print(f"x: {x}, y: {y}")
            break
    else:
        print(f"Image {img_name} not found in the JSON data.")

    image = unnormalize(image)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Ball")
    #remove pre process and normilization)
    plt.scatter(x, y, c='r', s=50)
    plt.show() 

def test_dataloader():
    """Test the dataloader by displaying images and labels."""
    for images, labels in dataloader:
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # Get a single image and label
            image = images[i]
            label = labels[i].item()
            print(labels[i])

            # Unnormalize the image
            unnormalized_img = unnormalize(image.unsqueeze(0))  # Add batch dim
            img = unnormalized_img.squeeze().permute(1, 2, 0).numpy()  # Convert to HWC

            # Convert to BGR format for OpenCV 
            img_bgr = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

            # Display the image
            plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Label: {label}")
            plt.show()
        
def test_collate_fn():
    """Test the collate function by creating a batch of images and labels and 
        checking the output of mean alighns with the label."""
    batch = {
            "regions": torch.stack([
                torch.ones(3, 224, 224),  # Positive region
                torch.zeros(3, 224, 224),  # Negative region
                torch.full((3, 224, 224), 3),  # positve region
                torch.full((3, 224, 224), 4), #negative region
                torch.full((3, 224, 224), 5), #positive region
                torch.full((3, 224, 224),6 ), #negative region
                torch.full((3, 224, 224), 7), #positive region
                torch.full((3, 224, 224), 8), #negative region

            ]),
            "labels": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        }
    
    randomized_images, randomized_labels = FoosballDataset.collate_fn([batch, batch])
    
    mean_to_label = {
    1.0: (1, "Positive region should have label 1"),
    0.0: (0, "Negative region should have label 0"),
    3.0: (1, "Positive region should have label 1"),
    4.0: (0, "Negative region should have label 0"),
    5.0: (1, "Positive region should have label 1"),
    6.0: (0, "Negative region should have label 0"),
    7.0: (1, "Positive region should have label 1"),
    8.0: (0, "Negative region should have label 0")
    }
    for region, label in zip(randomized_images, randomized_labels):
        region_mean = region.mean().item()
        expected = mean_to_label.get(region_mean)
        expected_label, error_message = expected
        assert label.item() == expected_label, error_message
    print ("Test collate passed")

def test_getRegionWithBall_createdData():
    """Test the getRegionWithBall with created data"""
    # Test parameters
    ball_exists = 1  # Ball is present
    x, y = 200, 200    # Ball coordinates
    regions = [i for i in range(16)]  # Simulate 16 regions (4x4 grid)
    region_height = 100
    region_width = 100

    # Call the method on the instance
    positive_region, region_index = dataset.getRegionWithBall(
        ball_exists, x, y, regions, region_height, region_width
    )

    # Assertions
    assert region_index == 10, f"Expected region_index to be 0, got {region_index}"
    assert positive_region == regions[10], f"Expected positive_region to be {regions[4]}, got {positive_region}"

    print("Test passed: getRegionWithBall correctly identified the region.")

def main():
    visulize_image()
    test_collate_fn()
    test_getRegionWithBall_createdData()
    test_dataloader()

if __name__ == "__main__":
    main()