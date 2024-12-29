import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

class SnoutNet(nn.Module):
    def __init__(self):
        """ Initialize the SnoutNet model. """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding=2)
        self.mp = nn.MaxPool2d(3, 4)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.relu = nn.ReLU()
        self.input_shape = (3, 227, 227)

    def forward(self, input_tensor: torch.Tensor):
        # print("Size of Input:", input_tensor.shape)
        X = self.conv1(input_tensor)
        X = F.relu(X)
        X = self.mp(X)
        # print("Shape After Layer 1:", X.shape)
        X = self.conv2(X)
        X = F.relu(X)
        X = self.mp(X)
        # print("Shape After Layer 2:", X.shape)
        X = self.conv3(X)
        X = F.relu(X)
        X = self.mp(X)
        # print("Shape After Layer 3:", X.shape)

        X = X.view(X.shape[0], -1)
        # print("Shape After Restructuring:", X.shape)
        
        X = self.fc1(X)
        X = F.relu(X)
        # print("Shape After FC1:", X.shape)
        X = self.fc2(X)
        X = F.relu(X)
        # print("Shape After FC2:", X.shape)
        X = self.fc3(X)
        # print("Shape After FC3:", X.shape)

        return X

if __name__ == "__main__":
    # Instantiate the model
    model = SnoutNet()

    # Create an input tensor with size (batch_size=1, channels=3, height=227, width=227)
    input_tensor = torch.randn(1, 3, 227, 227)
    # Pass the input through the model
    output = model(input_tensor)

