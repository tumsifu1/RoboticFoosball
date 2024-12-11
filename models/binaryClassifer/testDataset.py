import os
from torch.utils.data import DataLoader
import torch
import json
from torchvision import transforms
from FoosballDataset import FoosballDataset
import matplotlib.pyplot as plt
import cv2
json_path = "data/labelled_data1.json"
images_dir = "data/labelled_images_1"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle= True, collate_fn=FoosballDataset.collate_fn)
print(f"Dataset size: {len(dataset)}")

def unnormalize(image, mean, std):
    """Undo normalization for visualization."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean #tensor-wise math 



# Assuming `dataloader` is defined and ready

def test_dataloader(dataloader, mean, std):
    for images, labels in dataloader:
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # Get a single image and label
            image = images[i]
            label = labels[i].item()

            # Unnormalize the image
            unnormalized_img = unnormalize(image.unsqueeze(0), mean, std)  # Add batch dim
            img = unnormalized_img.squeeze().permute(1, 2, 0).numpy()  # Convert to HWC

            # Convert to BGR format for OpenCV
            img_bgr = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)

            # Display the image
            plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Label: {label}")
            plt.show()

def __main__():
    # Define mean and std for unnormalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_dataloader(dataloader, mean, std)

if __name__ == "__main__":
    __main__()