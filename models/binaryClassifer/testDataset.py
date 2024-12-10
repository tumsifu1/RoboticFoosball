import os
from torch.utils.data import DataLoader
import torch
import json
from torchvision import transforms
from FoosballDataset import FoosballDataset
import matplotlib.pyplot as plt
json_path = "data/labelled_data1.json"
images_dir = "data/labelled_images_1"

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Unnormalize function
def unnormalize(img, mean, std):
    img = img.clone()
    for c in range(3):  # Apply channel-wise
        img[c] = img[c] * std[c] + mean[c]
    return img

# Test the DataLoader
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

for batch_idx, (images, heights, widths) in enumerate(dataloader):
    image = images[0]  # Remove batch dimension
    unnormalized_img = unnormalize(image, mean, std)  # Undo normalization
    img = unnormalized_img.permute(1, 2, 0).numpy()  # Convert to HWC for plotting
    #img = image.permute(1, 2, 0).numpy() #no normilization
    plt.imshow(img)
    plt.title(f"Batch {batch_idx}")
    plt.axis('off')
    plt.show()
    break

