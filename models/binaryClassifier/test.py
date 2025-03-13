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
from torch.utils.data import random_split, DataLoader
from typing import Optional
from FoosballDataset import FoosballDataset
from models.binaryClassifier.model_resNet18Base import BinaryClassifier
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.tools.unnormalize import unnormalize

def displayImage(image, groundTruth,predicted ):
        # Unnormalize the image for visualization.
        # (Assumes image tensor is (B, C, 224, 224) normalized using ImageNet stats)
        unnorm_img = unnormalize(image)  # now in [0,1]

        
        img_np = unnorm_img[0].permute(1, 2, 0).cpu().numpy()
        # Clip to [0,1] then convert to [0,255] uint8.
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        

        plt.imshow(img_np)
        plt.title(f"Ground Truth: {groundTruth} | Predicted: {predicted}", fontsize=12, color='white', backgroundcolor='black')
        plt.show()
def saveConfusionMatrix(test_labels, test_preds, output_dir ):
    print(test_preds)
    # Compute and display confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix Test')
    cm_path = os.path.join(output_dir, f'confusion_matrix_test.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Test confusion matrix saved at: {cm_path}")

def test(**kwargs) -> None:
    for kwarg in kwargs:
        print(f"{kwarg}: {kwargs[kwarg]}")
    model = kwargs["model"]
    test_loader = kwargs["test"]
    device = kwargs["device"]
    output_dir = kwargs["output"]
    #testing
    test_labels = []
    test_preds = []
    print("starting testing")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            #confusion matrix
            pred = torch.round(torch.sigmoid(output)) #convert to binary 
            test_preds.extend(pred.cpu().squeeze().tolist())
            #print(test_preds)
            test_labels.extend(labels.cpu().tolist())

            displayImage(images, labels[0], pred[0])
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    saveConfusionMatrix(test_labels, test_preds , output_dir)
    print(f"Finished testing")

def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/binary_classifier')
    argParser.add_argument('-model', metavar='model', type=str, help='path to model', default='./src/weights/classifier.pth') 
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
    model.load_state_dict(torch.load(args.model))# Apply weights from training
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.BCEWithLogitsLoss()

    test(  
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

