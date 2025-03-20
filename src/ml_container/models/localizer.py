import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
class BallLocalization(nn.Module):
    def __init__(self, pretrained=False):
        super(BallLocalization, self).__init__()

        # Load MobileNetV3-Small as the feature extractor
        self.mobilenet = mobilenet_v3_small(pretrained=pretrained)

        # Remove the final classification layer
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[0].in_features, 256),  # Custom FC layer
            nn.ReLU(),
            nn.Linear(256, 2)  # Output raw (x, y) coordinates
        )

        # Initialize weights for new layers
        nn.init.xavier_uniform_(self.mobilenet.classifier[0].weight)
        nn.init.xavier_uniform_(self.mobilenet.classifier[2].weight)

    def forward(self, x):
        x = self.mobilenet(x)  # Pass through MobileNetV3-Small
        return x  # Output raw (x, y) coordinates

def main():
    model = BallLocalization(pretrained=True)
    dummy_input = torch.randn(1, 3, 2304//8, 1296//8)  # Example input
    output = model(dummy_input)
    print(f"Output (Raw x, y): {output}")

if __name__ == "__main__":
    main()