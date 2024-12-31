import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from typing import Optional
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDataset
from model import BallLocalization
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train(epochs: Optional[int] = 30, **kwargs) -> None:
    print("Starting training...")
    
    for kwarg in kwargs:
        print(f"{kwarg}: {kwargs[kwarg]}")
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    scheduler = kwargs["scheduler"]
    loss_function = kwargs["loss_function"]
    train_loader = kwargs["train"]
    test_loader = kwargs["test"]
    device = kwargs["device"]
    output_dir = kwargs["output"]

    #lists for trianign and val loss
    losses_train = []
    losses_val = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for image, x,y, _ in train_loader:
            images = image
            labels = torch.stack((x, y), dim=1) #stack x and y coordinates
            # print(batch)
            #print(f"images: {images}")
            #print(f"Labels: {labels}")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images) 
            
            loss = loss_function(output, labels) 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader) #average loss over the epoch
        losses_train.append(train_loss)
    
        model.eval()
        val_loss = 0 
        with torch.no_grad():
            for image, x,y, _ in test_loader:
                images = image
                labels = torch.stack((x, y), dim=1) #stack x and y coordinates
                images, labels = images.to(device), labels.to(device)

                output = model(images)
        
                #compute loss 
                val_loss += loss_function(output,  labels.unsqueeze(1).float())
    
        val_loss /= len(test_loader)
        val_loss = val_loss.item()
        losses_val.append(val_loss)
        scheduler.step(val_loss)

        file_path = f"{output_dir}/stats.txt"

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                pass

        with open(file_path, "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} ")
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {accuracy:.5f}")

        torch.save(model.state_dict(), f"{output_dir}/model.pth")

        # Plot and save loss plot
        plt.figure(figsize=(12, 7))
        plt.grid(True)
        plt.plot(losses_train, label='Train Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{output_dir}/loss_plot.png")
        plt.close()
        
    print("Training complete.")
def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/ball_Localization')
    args = argParser.parse_args()

    #random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    #make sure output folder exists 
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #check if cuda is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    #load the dataset
    images_dir = args.images
    json_path = args.labels
    train_dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=None)
    test_dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=None)

    #load the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=FoosballDataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=FoosballDataset.collate_fn)
    
    #load the model
    model = BallLocalization()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.MSELoss(size_average=None, reduce=None, reduction='mean')


    train(  
        epochs=args.epoch, 
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,    
        loss_function=loss_function, 
        train=train_dataloader, 
        test=test_dataloader, 
        device=device,
        output=args.output,
        )

if __name__ == "__main__":
    main()

