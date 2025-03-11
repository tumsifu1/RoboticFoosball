import sys
import os

# Move up two levels to add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)


import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from typing import Optional
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer
#from models.ballLocalization.model_snoutNetBase import BallLocalization
from models.ballLocalization.model_mobileNetV3Base import BallLocalization
import numpy as np
import random

# Create a directory for saving plots
output_dir = "gradient_plots"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store gradient norms over epochs
gradient_history = {}


def track_and_plot_gradients(epoch, model, outputs, save_path="gradient_plots/gradients_epoch"):
    global gradient_history

    # Track gradient norms
    layer_names = []
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            layer_names.append(name)
            grad_norms.append(grad_norm)
            
            # Store gradients over epochs
            if name not in gradient_history:
                gradient_history[name] = []
            gradient_history[name].append(grad_norm)

    # Plot only a subset of layers for clarity (every Nth layer)
    N = max(1, len(layer_names)//20)  # Adjusted for better readability
    selected_layers = layer_names[::N]
    selected_grads = grad_norms[::N]

    # Plot Gradient Norms per Layer
    plt.figure(figsize=(16,14))
    plt.plot(selected_layers, selected_grads, marker="o", linestyle="-", label=f"Epoch {epoch+1}")
    plt.yscale("log")  # Log scale for better visualization
    plt.xlabel("Layers")
    plt.ylabel("Gradient Norm")
    plt.xticks(rotation=45, fontsize=8)
    plt.title("Gradient Norms per Layer")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"{save_path}_{epoch+1}.png")
    plt.close()

    #Print some key gradients
    print(f"Epoch {epoch+1} - Sample Gradients:")
    for i in range(0, len(layer_names), max(1, len(layer_names) // 5)):
        print(f"  {layer_names[i]}: {grad_norms[i]}")

    return grad_norms

def train_one_epoch( epoch, model, optimizer, scheduler,loss_function, train_loader, test_loader, 
                    val_loader, device, output_dir):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        image, x, y, x_pre227scaling, y_pre227scaling = batch[:5]
        print(f"Batch Number : {i+1}")
        print(f"Inputs: {x[0]}, {y[0]}")
        image, x, y = image.to(device), x.to(device), y.to(device)
        
        #actual_x, actual_y = x, y
        #predicted_x, predicted_y = output[0, 0].item(), output[0, 1].item()
        #print(f"Label vs Prediction Before Rescale: Actual ({actual_x}, {actual_y}), Predicted ({predicted_x}, {predicted_y})")
        
        images = image
        labels = torch.stack((x, y), dim=1) #stack x and y coordinates
        # print(batch)
        #print(f"images: {images}")
        #print(f"Labels: {labels}")
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        output = model(images) 
        print(output.shape)
        pred_x, pred_y = output[0, 0].item(), output[0, 1].item()
        #print(f"output x: {pred_x},  y: {pred_y}")
        
        loss = loss_function(output, labels) 
        train_loss+=loss

        # if epoch < 5: #NOTE:Doing nothing right now
        #     for name, param in model.named_parameters():
        #         if "classifier" not in name:
        #             param.requires_grad = True
        # else:
        #     for param in model.parameters():
        #         param.requires_grad = True
        
        loss.backward() #get the gradients
        track_and_plot_gradients(epoch, model, output_dir, save_path="gradient_plots/gradients_epoch")

        #Clip gradient norms and set max norm to 5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    train_loss /= len(train_loader) #average loss over the epoch
    return train_loss 
def validate_one_epoch( model, optimizer, scheduler,loss_function, train_loader, test_loader, 
                    val_loader, device, output_dir):
    model.eval()
    val_loss = 0 
    mse_total = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            image, x, y = batch[:3]
            images = image
            labels = torch.stack((x, y), dim=1) #stack x and y coordinates
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            val_loss += loss_function(output, labels).item()
            mse = ((output - labels) ** 2).mean().item()
            mse_total += mse
            total += 1
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    return val_loss, mse_total,total

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

    #lists for trianign and val loss
    losses_train = []
    losses_val = []
    counter = 0
    best_val_loss = float("inf")
    patience = 10 #number of epochs with allowed with no improvement
    for epoch in range(epochs):
        
        #train model and add loss values to array
        train_loss = train_one_epoch(epoch, model, optimizer, scheduler,loss_function, train_loader, test_loader, 
                    val_loader, device, output_dir)
        losses_train.append(train_loss.item())
        
        #validate mdoel
        val_loss, mse_total,total = validate_one_epoch(model, optimizer, scheduler,loss_function, train_loader, test_loader, 
                    val_loader, device, output_dir)
        
        #calculate and save stats
        mse_average = mse_total / total 
        losses_val.append(val_loss)

        file_path = f"{output_dir}/stats.txt"
        with open(file_path, "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | MSE: {mse_average:.5f} | lr: {optimizer.param_groups[0]['lr']}\n")
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | MSE: {mse_average:.5f}")
        torch.save(model.state_dict(), f"{output_dir}/model_epoch_{epoch+1}.pth")

        #save best model
        if val_loss <= min(losses_val):
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # Plot and save loss plot
        plt.figure(figsize=(12, 7))
        plt.grid(True)
        plt.plot(losses_train, label='Train Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{output_dir}/loss_plot.png")
        plt.close()

        if val_loss<=best_val_loss:
            best_val_loss = val_loss 
            counter = 0
        else:
            counter+=1
            if counter >= patience:
                print("Stopped training val loss is not improving")
                break


    print("Training complete.")

# def init_weights(m):
#     """ Initialize default weights for the model. """
#     if type(m) == nn.Linear or type(m) == nn.Conv2d:
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def digest_arguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=100, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, help='path to images directory (default: ./data/image_data/images)', default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, help='path to labels directory', default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    argParser.add_argument('-output', metavar='output', type=str, help='output directory', default='./output/ball_Localization')
    argParser.add_argument('-num_workers', metavar='num_workers', type=int, help='number of workers for dataloader', default=2)

    return argParser.parse_args()
def main():

    args = digest_arguments()
    #test, train and val paths
    val_images = "./data/val/images"
    val_labels = "./data/val/labels/labels.json"
    test_images = "./data/test/images"
    test_labels = "./data/test/labels/labels.json"
    train_images = "./data/train/images"
    train_labels = "./data/train/labels/labels.json"
    number_workers = args.num_workers
    #random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)
        
    #make sure output folder exists 
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #check if cuda is available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    #load the dataset (done in the dataset class)
    #images_dir = args.images
    #json_path = args.labels
    
    # Create new dataset instances for each split
    train_dataset = FoosballDatasetLocalizer(json_path=train_labels, images_dir=train_images, transform=None, train=True)
    val_dataset = FoosballDatasetLocalizer(json_path=val_labels, images_dir=val_images, transform=None, train=False)
    test_dataset = FoosballDatasetLocalizer(json_path=test_labels, images_dir=test_images, transform=None, train=False)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,num_workers = number_workers, collate_fn=FoosballDatasetLocalizer.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers = number_workers, collate_fn=FoosballDatasetLocalizer.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers = number_workers, collate_fn=FoosballDatasetLocalizer.collate_fn)

    #load the model
    model = BallLocalization()
    #model.load_state_dict(torch.load("output/ball_Localization/best_model.pth"))
    model.apply(init_weights)
    model.to(device)
    #optimizer = optim.Adam([
    #    {'params': model.model.features.parameters(), 'lr': 1e-4},
    #   {'params': model.model.classifier.parameters(), 'lr': 1e-3},
    #], weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_function = nn.HuberLoss(delta=22.7)

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