import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split, DataLoader
from typing import Optional
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer
from model import BallLocalization
import numpy as np
import random

def train(epochs: Optional[int] = 30, **kwargs) -> None:
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
        for image, x,y in test_loader:
            images = image
            labels = torch.stack((x, y), dim=1) #stack x and y coordinates
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            #confusion matrix
            if output.size(0) != labels.size(0):
                output = output[:labels.size(0)]
            mse = ((output - labels) ** 2).mean().item()
            test_mse += mse
            total += 1

    test_mse /= total
    file_path = f"{output_dir}/test_stats.txt"
    with open(file_path, "a") as f:
        f.write(f" MSE: {test_mse:.5f}\n")

    print(f"Test MSE: {test_mse}")
    print(f"Finished testing")

def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
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

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=FoosballDatasetLocalizer.collate_fn)

    #load the model
    model = BallLocalization()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.MSELoss(reduction='mean')

    train(  
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

