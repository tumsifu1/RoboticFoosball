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
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

def unnormalize(image, mean, std):
    """Undo normalization for visualization."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean

# Define mean and std for unnormalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Assuming `dataloader` is defined and ready
for batch in dataloader:
    if "relative_coords" in batch:  # Train mode
        images = batch["regions"]  # Stack of two regions (labeled and random)
        labels = batch["labels"]  # Labels for the two regions
        relative_coords = batch["relative_coords"]  # Tensor of coordinates (B, 2)

        # Process each sample in the batch
        for i in range(images.shape[0]):
            labeled_image = images[i, 0]  # The labeled region
            random_image = images[i, 1]  # The random region


            ball_x, ball_y = relative_coords[0].tolist()[0], relative_coords[1].tolist()[0]  # Convert tensor to list and unpack

            # Unnormalize the labeled image
            unnormalized_labeled_img = unnormalize(labeled_image, mean, std)
            lbl_img = unnormalized_labeled_img.permute(1, 2, 0).numpy()

            # Overlay the ball coordinates on the labeled region
            lbl_img_bgr = cv2.cvtColor((lbl_img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
            cv2.circle(lbl_img_bgr, (int(ball_x), int(ball_y)), 10, (255, 0, 0), -1)

            # Display labeled image
            plt.imshow(cv2.cvtColor(lbl_img_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Labeled Region with Ball (Sample {i})")
            plt.show()

            # Unnormalize and display random region
            unnormalized_random_img = unnormalize(random_image, mean, std)
            rnd_img = unnormalized_random_img.permute(1, 2, 0).numpy()

            plt.imshow(rnd_img)
            plt.axis("off")
            plt.title(f"Random Region (Sample {i})")
            plt.show()

    else:  # Test mode
        regions = batch["regions"]  # All regions in the test image
        labels = batch["labels"]  # Corresponding labels for all regions

        for i, (region, label) in enumerate(zip(regions, labels)):
            unnormalized_region = unnormalize(region, mean, std)
            region_img = unnormalized_region.permute(1, 2, 0).numpy()

            # Display each region
            plt.imshow(region_img)
            plt.axis("off")
            plt.title(f"Region {i} - Label: {label.item()}")
            plt.show()

    break  # Only process the first batch