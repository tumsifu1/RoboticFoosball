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
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from typing import Optional
from FoosballDataset import FoosballDataset
from model import BinaryClassifier
import numpy as np
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train(epochs: Optional[int] = 1, **kwargs) -> None:
    print("Starting training...")
    
    for kwarg in kwargs:
        print(f"{kwarg}: {kwargs[kwarg]}")
    model = kwargs["model"]
    optimizer = kwargs["optimizer"]
    scheduler = kwargs["scheduler"]
    loss_function = kwargs["loss_function"]
    train_loader = kwargs["train"]
    test_loader = kwargs["test"]
    val_loader = kwargs["val"]
    device = kwargs["device"]
    output_dir = kwargs["output"]

    #lists for training and val loss
    losses_train = []
    losses_val = []
    patience = 10
    counter = 0
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        all_labels = []
        all_preds = []
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            images, labels = batch
            print(f"batch: {i}")
            # print(batch)
            #print(f"images: {images}")
            #print(f"Labels: {labels}")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images) 
            
            loss = loss_function(output, labels.unsqueeze(1).float()) #compute loss and add dimension to labels
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader) #average loss over the epoch
        losses_train.append(train_loss)

        #validation

        #validation
        model.eval()
        val_loss = 0 
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
        
                #compute loss 
                val_loss += loss_function(output,  labels.unsqueeze(1).float())

                #compute accuracy
                pred = torch.round(torch.sigmoid(output)) #convert to binary 
                correct += (pred.squeeze() == labels).sum().item() #sum the correct predictions
                total += labels.size(0)
                all_preds.extend(pred.cpu().squeeze().tolist())
                all_labels.extend(labels.cpu().tolist())
                
        val_loss /= len(val_loader)
        losses_val.append(val_loss)

        #scheduler step
        scheduler.step(val_loss)
        #save stats to output folder
        accuracy = 100* correct/total 
        file_path = f"{output_dir}/stats.txt"
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                pass

        with open(file_path, "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {accuracy:.5f}\n")
        
        print(f"Correct: {correct}, Total: {total}")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Acc: {accuracy:.5f}")
        
         #save mode weights
        torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch+1}.pth")

        if val_loss <= min(losses_val):
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

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
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        # Compute and display confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        cm_path = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved at: {cm_path}")

        if val_loss<=best_val_loss:
            best_val_loss = val_loss 
            counter = 0
        else:
            counter+=1
            if counter >= patience:
                print("Stopped training val loss is not improving")
                break
    
    print("Training complete.")


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=32)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/binary_classifier')
    argParser.add_argument('-num_workers', metavar='num_workers', type=int, help='number of workers for dataloader', default=0)
    args = argParser.parse_args()

    #random seed for reproducibility
    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = 'cuda'
    number_workers = args.num_workers
        #test, train and val paths
    val_images = "./data/val/images"
    val_labels = "./data/val/labels/labels.json"
    test_images = "./data/test/images"
    test_labels = "./data/test/labels/labels.json"
    train_images = "./data/train/images"
    train_labels = "./data/train/labels/labels.json"

    #make sure output folder exists 
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #check if cuda is available
    print(f'Using: {device}')

    #load the dataset
    #images_dir = args.images
    #json_path = args.labels

    # Create new dataset instances for each split
    train_dataset = FoosballDataset(json_path=train_labels, images_dir=train_images, transform=None, train=True)
    val_dataset = FoosballDataset(json_path=val_labels, images_dir=val_images, transform=None, train=False)
    test_dataset = FoosballDataset(json_path=test_labels, images_dir=test_images, transform=None, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=number_workers ,collate_fn=FoosballDataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers= number_workers, collate_fn=FoosballDataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers = number_workers, collate_fn=FoosballDataset.collate_fn)

    #load the model
    model = BinaryClassifier()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.BCEWithLogitsLoss()

    train(  
        epochs=args.epoch, 
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler,    
        loss_function=loss_function, 
        train=train_dataloader, 
        val = val_dataloader,
        test=test_dataloader, 
        device=device,
        output=args.output,
        )

if __name__ == "__main__":
    main()




