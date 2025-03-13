import os
from torch.utils.data import DataLoader
import torch
import json
from torchvision import transforms
from models.binaryClassifier.FoosballDataset import FoosballDataset
from models.binaryClassifier.model_resNet18Base import BinaryClassifier 
from src.tools.unnormalize import unnormalize
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
json_path = "data/train/labels/labels.json"
images_dir = "data/train/images"
from torchvision import transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle= False, collate_fn=FoosballDataset.collate_fn)
print(f"Dataset size: {len(dataset)}")

def visulize_image(): #You must comment out to tensor in the preprocess
    """Visualize a full image from the dataset."""
    img_name = "img_0.jpg"
    img_path = "data/images/" + img_name
    image = Image.open(img_path).convert('RGB')
    #1102,
    #646
    ball_exists = True
    x = 0
    y = 0
    with open(json_path, 'r') as f:
        data = json.load(f)

    ballData = data[img_name]

    x = ballData["x"]
    y = ballData["y"]

    #image = unnormalize(image)
    image_array = np.array(image)

    plt.imshow(image_array)
    plt.axis("off")
    plt.title(f"Full Image")
    #remove pre process and normilization)
    plt.scatter(x, y, c='r', s=25)
    plt.show() 

def slice_images():
    img_name = "img_80.jpg"
    img_path = "data/images/" + img_name
    image = Image.open(img_path).convert('RGB')

    transform = transforms.ToTensor()
    image = transform(image)

    dataset = FoosballDataset(images_dir="data/images", json_path="data/labels/labels.json", transform=None, train=False)
    regions = dataset.breakImageIntoRegions(image)  # Get 64 sliced regions

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i >= len(regions):
            ax.axis("off")
            continue

        img_np = regions[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.axis("off")

    # Save the figure
    plt.savefig("sliced_images.png", dpi=300, bbox_inches='tight')
    plt.show()

def display_images(device): 
    img_name = "img_80.jpg"
    img_path = "data/images/" + img_name
    image = Image.open(img_path).convert('RGB')
    
    transform = transforms.ToTensor()
    image = transform(image)
    
    model = BinaryClassifier()
    model.load_state_dict(torch.load("./src/weights/classifier.pth", map_location=device))  # Load weights
    model.to(device)
    model.eval()
    
    dataset = FoosballDataset(images_dir="data/images", json_path="data/labels/labels.json", transform=None, train=False)
    regions = dataset.breakImageIntoRegions(image)
    
    all_probs = []  # Store all prediction probabilities

    with torch.no_grad():
        for region in regions:
            region = region.unsqueeze(0).to(device)  # Add batch dimension
            output = model(region)
            prob = torch.sigmoid(output).item()  # Convert to scalar
            all_probs.append(prob)

    # Convert list to tensor for argmax
    all_probs = torch.tensor(all_probs)
    max_index = torch.argmax(all_probs).item()

    # Create binary predictions (only one highest gets 1)
    pred = torch.zeros_like(all_probs)
    pred[max_index] = 1

    #did not unom
    unnorm_images = regions 

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i >= len(regions):
            ax.axis("off")
            continue

        img_np = unnorm_images[i].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.axis("off")

        # Highlight only the highest probability region
        if pred[i] == 1:
            ax.spines['top'].set_color('red')
            ax.spines['bottom'].set_color('red')
            ax.spines['left'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)

        ax.set_title(f"Pred: {int(pred[i])}", fontsize=6, color='white', backgroundcolor='black')
    plt.savefig("prediction.png", dpi=300, bbox_inches='tight')
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
    #visulize_image()
    #test_collate_fn()
    #test_getRegionWithBall_createdData()
    slice_images()
    display_images( "cpu")
    #test_dataloader()

if __name__ == "__main__":
    main()