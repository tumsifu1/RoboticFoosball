import torch
from models.snoutNet.model import SnoutNet
import torch.nn as nn
import torch.nn.functional as F

class BallLocalization(SnoutNet):
    def __init__(self):
        super(BallLocalization, self).__init__()
        self.input_shape = (3, 2304, 1296)
        
        # Dynamically compute the flattened size
        flattened_size = self._get_flattened_size(self.input_shape)
        print(f"Dynamically computed flattened size: {flattened_size}")
        
        # Redefine fully connected layers with the correct size
        self.fc1 = nn.Linear(13824, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def _get_flattened_size(self, input_shape):
        """Calculate the flattened size dynamically based on input shape."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Dummy input with batch size 1
            x = self.mp(F.relu(self.conv1(x)))
            x = self.mp(F.relu(self.conv2(x)))
            x = self.mp(F.relu(self.conv3(x)))
            print(f"Shape after conv3 + mp in _get_flattened_size: {x.shape}")
            return x.numel()  # Correct flattened size

def main():
    model = BallLocalization()
    dummy_input = torch.randn(1,3, 2304, 1296)  # Batch size of 64
    output = model(dummy_input)
    print(output.size())
if __name__ == "__main__":
    main()
