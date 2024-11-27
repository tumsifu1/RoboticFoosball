import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import BinaryClassifier
from typing import Optional

def train(epochs: Optional[int] = 30, **kwargs) -> None:
    
    print("Training Parameters")
    for kwarg in kwargs:
        print(f"{kwarg} = {kwargs[kwarg]}")

    model = kwargs['model']
    model.train()
    optimizer = kwargs['optimizer']
    scheduler = kwargs['scheduler']
    loss_function = kwargs['loss_function']
    #dataloader stuff will go here (Need to discuss this)
    

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", type=int, default=30)
    argParser.add_argument('-i', metavar='images_directory', type=str, help='path to images directory (default: ./data/images)', default='./data/images')
    argParser.add_argument('-l', metavar='labels', type=str, help='path to labels directory', default='./data/labels')
    argParser.add_argument('-b', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    argParser.add_argument('-o', metavar='output', type=str, help='output directory', default='./output/binary_classifier')
    args = argParser.parse_args()

    #make sure output folder exists 
    if not os.path.exists('./output/binary_classifier'):
        os.makedirs('./output/binary_classifier')
    
    #check if cuda is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    #dataloader stuff will go here (Need to discuss this)

    #load the model\
    model = BinaryClassifier()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.BCELoss()


if __name__ == "__main__":
    main()




