import torch
from models.snoutNet.model import SnoutNet
import torch.nn as nn
import torch.nn.functional as F

class BallLocalization(SnoutNet):
    def __init__(self):

        super(BallLocalization, self).__init__()

        nn.init.xavier_uniform_(self.fc3.weight)

    def _get_flattened_size(self, input_shape):
        """Calculate the flattened size dynamically based on input shape."""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Dummy input with batch size 1
            x = self.mp(F.relu(self.conv1(x)))
            x = self.mp(F.relu(self.conv2(x)))
            x = self.mp(F.relu(self.conv3(x)))
            print(f"Shape after conv3 + mp in _get_flattened_size: {x.shape}")
            return -123412
            return x.numel()  # Correct flattened size

    def forward(self, x):
        model_out = super().forward(x)
        print(f"Output directly from model: {model_out[0]}")
        #out = torch.sigmoid(model_out)
        #out = torch.sigmoid(model_out)
        #out = (torch.tanh(model_out) + 1) / 2
        #print(f" output after sigmoid: {out}")

        return model_out

def main():
    model = BallLocalization()
    dummy_input = torch.randn(1,3, 227, 277)  # Batch size of 64
    output = model(dummy_input)
    print(output.size())
if __name__ == "__main__":
    main()
