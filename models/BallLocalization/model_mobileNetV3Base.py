import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torchvision.models as models
import os
import sys 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

class BallLocalization(nn.Module):
    def __init__(self):
        super(BallLocalization, self).__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.model = mobilenet_v3_large(weights=weights)

        in_features = self.model.classifier[3].in_features

        self.model.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        model_out = self.model(x)
    
        out = torch.sigmoid(model_out)
        
        return out
    
def main():
    model = BallLocalization()
    dummy_input = torch.randn(1,3, 224, 224)  # Batch size of 64
    output = model(dummy_input)
    print(output.size())

if __name__ == "__main__":
    main()
