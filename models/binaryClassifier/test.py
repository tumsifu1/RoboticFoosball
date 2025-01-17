import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split, DataLoader
from typing import Optional
from FoosballDataset import FoosballDataset
from model import BinaryClassifier
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    test_labels = []
    test_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            #confusion matrix
            pred = torch.round(torch.sigmoid(output)) #convert to binary 
            test_preds.extend(pred.cpu().squeeze().tolist())
            test_labels.extend(labels.cpu().tolist())

        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        # Compute and display confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix Test')
        cm_path = os.path.join(output_dir, f'confusion_matrix_test.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Test confusion matrix saved at: {cm_path}")

    print(f"Finished testing")

def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/binary_classifier')
    argParser.add_argument('-model', metavar='model', type=str, help='path to model', default='./output/binary_classifier/best_model.pth')
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
    
    test_dataset = FoosballDataset(json_path=test_labels, images_dir=test_images, transform=None, train=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=FoosballDataset.collate_fn)

    #load the model
    model = BinaryClassifier()
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

