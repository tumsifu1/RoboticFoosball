import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from typing import Optional
from FoosballDataset import FoosballDataset
from model import BinaryClassifier
import numpy as np
import random

def train(epochs: Optional[int] = 30, **kwargs) -> None:
    
    print("Training Parameters")
    for kwarg in kwargs:
        print(f"{kwarg} = {kwargs[kwarg]}")

    model = kwargs['model']
    model.train()
    optimizer = kwargs['optimizer']
    scheduler = kwargs['scheduler']
    loss_function = kwargs['loss_function']

    losses_train = [] #array to store training losses for plotting
    losses_val = [] #array to store validation lossesc for plotting

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        for batch_idx, (data, label) in enumerate(kwargs['train']):
            data, label = data.to(kwargs['device']), label.to(kwargs['device'])
            optimizer.zero_grad()
            output = model(data)
            label = label.unsqueeze(1) #add a dimension to match the output shape
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(kwargs['train'])
        losses_train.append(train_loss)
        model.eval()
        correct = 0

        with torch.no_grad():
            for data, label in kwargs['test']:
                data, label = data.to(kwargs['device']), label.to(kwargs['device'])
                output = model(data)
                label = label.unsqueeze(1) #add a dimension to match the output shape
                val_loss += loss_function(output, label).item()
                pred = output.round()
                correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss /= len(kwargs['test'])
        losses_val.append(val_loss)

        accuracy = 100. * correct / len(kwargs['test'].dataset)
        print(f"\nTest set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(kwargs['test'].dataset)} ({accuracy:.0f}%)\n")
        scheduler.step(val_loss)
        print(f'Saving Weights to {kwargs["output"]}')
        torch.save(model.state_dict(), os.path.join(kwargs['output'], 'model.pth'))

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.grid(True)
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(kwargs['output'], 'loss.png'))

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-image', metavar='images_directory', type=str, help='path to images directory (default: ./data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/binary_classifier')
    args = argParser.parse_args()

    #random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    #make sure output folder exists 
    if not os.path.exists(args.output):
        os.makedirs(args.oput)

    #check if cuda is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    # Define transformations
    #translation
    #zooming
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #load the dataset
    images_dir = args.image
    json_path = args.labrels
    train_dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=train_transform)
    test_dataset = FoosballDataset(json_path=json_path, images_dir=images_dir, transform=test_transform)

    #load the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)

    #load the model
    model = BinaryClassifier()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.BCEWithLogitsLoss() #mean squared for position 

    train(  
            epochs=args.e, 
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler,    
            loss_function=loss_function, 
            train=train_dataloader, 
            test=test_dataloader, 
            device=device,
            output=args.o
            )

if __name__ == "__main__":
    main()




