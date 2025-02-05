import sys
import os

# Get the absolute path of the current script's directory
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split, DataLoader
from typing import Optional
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer
from models.ballLocalization.model_mobileNetV3Base import BallLocalization
from src.tools import unnormalize
import numpy as np
import random

def test(epochs: Optional[int] = 30, **kwargs) -> None:
    print("Starting training...")
    
    for kwarg in kwargs:
        print(f"{kwarg}: {kwargs[kwarg]}")
    model = kwargs["model"]
    test_loader = kwargs["test"]
    device = kwargs["device"]
    output_dir = kwargs["output"]

    print("starting testing")

    #testing
    test_mse = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for image, x, y in test_loader:
            # Move tensors to device
            image, x, y = image.to(device), x.to(device), y.to(device)

            # Forward pass
            output = model(image)

            print(f"Output Shape: {output.shape}")  # Should be (batch_size, 2)

            # Extract single instance from batch
            img = image[0].permute(1, 2, 0).cpu().numpy()  # Convert to NumPy (H, W, C)
            actual_x, actual_y = x[0].cpu().item(), y[0].cpu().item()  # Extract ground truth
            predicted_x, predicted_y = output[0, 0].cpu().item(), output[0, 1].cpu().item()  # Extract predictions

            print(f"Label vs Prediction: Actual ({actual_x}, {actual_y}), Predicted ({predicted_x}, {predicted_y})")

            # Plot the image and points
            plt.imshow(img)
            plt.scatter([actual_x], [actual_y], color='red', marker='x', label='Actual')  # Actual point
            plt.scatter([predicted_x], [predicted_y], color='blue', marker='o', label='Predicted')  # Predicted point
            plt.legend()
            plt.show()  # Display the image

              # Only visualize one sample for nowhe loop after processing the first image

def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=1)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/ball_Localization')
    argParser.add_argument('-model', metavar='model', type=str, help='path to model', default='./output/ball_Localization/best_model.pth')
    args = argParser.parse_args()

    test_images = "./data/test/images"
    test_labels = "./data/test/labels/labels.json"
        
    #make sure output folder exists 
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #check if cuda is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')
    
    test_dataset = FoosballDatasetLocalizer(json_path=test_labels, images_dir=test_images, transform=None, train=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, collate_fn=FoosballDatasetLocalizer.collate_fn)

    #load the model
    model = BallLocalization()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.MSELoss(reduction='mean')

    test(  
        epochs=args.epoch, 
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,    
        loss_function=loss_function, 
        test=test_dataloader, 
        device=device,
        output=args.output,
        )

if __name__ == "__main__":
    main()

