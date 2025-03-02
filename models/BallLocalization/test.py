import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.nn.functional as F
# Import your dataset and model
from models.ballLocalization.FoosballDatasetLocalizer import FoosballDatasetLocalizer
from models.ballLocalization.model_snoutNetBase import BallLocalization
#from models.ballLocalization.model_mobileNetV3Base import BallLocalization

min_distance,  max_distance, sum_distance, sum_squared_distance = float('inf'), float('-inf'), 0, 0,
# Define a custom unnormalize function
def unnormalize_tensor(tensor,mean = [0.1249, 0.1399, 0.1198], std=[0.1205, 0.1251, 0.1123]):
    """
    Undo the normalization for a tensor image.
    Expects input tensor to be (B, C, H, W) or (C, H, W) in float.
    Returns a tensor in the range [0, 1].
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    # Create mean and std tensors
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    unnorm = tensor * std + mean
    return unnorm

def rescale_coordinates( old_x, old_y, region_width, region_height):
    # First, convert the pixel coordinate within the region to a normalized value [0,1]
    print(f"old x: {old_x}")
    print(f"old y {old_y}")
    new_x = old_x* region_width
    new_y = old_y* region_height
    # Then, scale that to the 224x224 space

    return new_x, new_y

def rescale_image(image_tensor, target_size):
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    rescaled = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
    return rescaled

def displayImage(image, scaled_x, scaled_y, actual_x,actual_y,predicted_x,predicted_y):
        # Unnormalize the image for visualization.
        # (Assumes image tensor is (B, C, 224, 224) normalized using ImageNet stats)
        unnorm_img = unnormalize_tensor(image)  # now in [0,1]

        # rescale from 227x224 back to the original region size.
        #original_region_size = (FoosballDatasetLocalizer.get_region_height(), FoosballDatasetLocalizer.get_region_width())  # (height, width)
        
        #rescaled_img = rescale_image(unnorm_img, (227,227))
        
        img_np = unnorm_img[0].permute(1, 2, 0).cpu().numpy()
        # Clip to [0,1] then convert to [0,255] uint8.
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        print(f"Label vs Prediction Normalize: Ground Truth ({actual_x}, {actual_y}), Predicted ({predicted_x}, {predicted_y})")
        
        # Rescale coordinates from the 224 space back to the original region dimensions.
        #rescaled_pred_x, rescaled_pred_y = rescale_coordinates(predicted_x, predicted_y, region_width=FoosballDatasetLocalizer.get_region_width(), region_height=FoosballDatasetLocalizer.get_region_height())
        #gtruth_not_normalized_x, gtruth_not_normalized_y = rescale_coordinates(x, y, region_width=FoosballDatasetLocalizer.get_region_width(), region_height=FoosballDatasetLocalizer.get_region_height())
        #print(f"Label vs Prediction Full size: Ground Truth ({gtruth_not_normalized_x}, {gtruth_not_normalized_y}), Predicted ({rescaled_pred_x}, {rescaled_pred_y})")
        
        # Plot the image and overlay the predicted and actual coordinates.
        plt.imshow(img_np)
        plt.scatter([scaled_x], [scaled_y], color='red', marker='x', label='Actual')
        plt.scatter([predicted_x], [predicted_y], color='blue', marker='o', label='Predicted')
        plt.legend()
        plt.show()

def calculate_mse(x,y,pred_x,pred_y):
        global sum_distance
        global sum_squared_distance 
        global min_distance
        global max_distance
        # Calculate Euclidean distances between predicted and actual points
        distances = torch.sqrt((x - pred_x)**2 + (y - pred_y)**2)

        min_distance = min(min_distance, torch.min(distances).item()) 
        max_distance = max(max_distance, torch.max(distances).item())  
        sum_distance += torch.sum(distances).item() 
        sum_squared_distance += torch.sum(distances**2).item() 


def test(epochs: int = 30, **kwargs) -> None:
    print("Starting testing...")
    model = kwargs["model"]
    test_loader = kwargs["test"]
    device = kwargs["device"]
    output_dir = kwargs["output"]

    model.eval()
    num_images = len(test_loader) #TODO is testloader the len?
    with torch.no_grad():
        for image, scaled_x, scaled_y, gtruth_not_scaledd_x, gtruth_not_scaled_y, _ in test_loader:

            # Forward pass through the model using the normalized image
            output = model(image)
            print(f"Output Shape: {output.shape}")  # Expected: (batch_size, 2)
            
            # Extract the first instance from the batch.
            # NOTE: Ensure that the coordinates (x, y) are in the 224-space if that’s how they were trained.
            actual_x, actual_y = scaled_x, scaled_y

            pred_x,pred_y = output[0, 0].item(), output[0, 1].item()
            #pred_x, pred_y = rescale_coordinates(pred_x, pred_y, FoosballDatasetLocalizer.get_region_width(), FoosballDatasetLocalizer.get_region_height())
            #actual_x, actual_y = rescale_coordinates(scaled_x, scaled_y, FoosballDatasetLocalizer.get_region_width(), FoosballDatasetLocalizer.get_region_height())
            print(f"inputs:{scaled_x, scaled_y}")
            # Move tensors to the device
            image, x, y = image.to(device), scaled_x.to(device), scaled_y.to(device)

            


            displayImage(image, actual_x, actual_y, actual_x,actual_y,pred_x,pred_y)
            calculate_mse(x,y,pred_x,pred_y)

    mean_distance = torch.tensor(sum_distance / num_images)
    std_dev = torch.sqrt((sum_squared_distance / num_images) - (mean_distance**2))


    print(f"Testing Completed:Minimum Distance: {min_distance}\nMaximum Distance: {max_distance}\nAverage Distance: {mean_distance}\nStandard Deviation: {std_dev}")


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-epoch", type=int, default=30, help="number of epochs")
    argParser.add_argument('-images', metavar='images_directory', type=str, default='./data/images')
    argParser.add_argument('-labels', metavar='labels', type=str, default='./data/labels/labels.json')
    argParser.add_argument('-batch', metavar='batch_size', type=int, default=1)
    argParser.add_argument('-output', metavar='output', type=str, default='./output/ball_Localization')
    argParser.add_argument('-model', metavar='model', type=str, default='output/ball_Localization/best_model_mse11_sd10.pth')
    args = argParser.parse_args()

    test_images = "./data/test/images"
    test_labels = "./data/test/labels/labels.json"
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')
    
    test_dataset = FoosballDatasetLocalizer(json_path=test_labels, images_dir=test_images, transform=None, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, collate_fn=FoosballDatasetLocalizer.collate_fn)

    model = BallLocalization()
    print(args.model)
    model.load_state_dict(torch.load(args.model))# Apply weights from training
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
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
