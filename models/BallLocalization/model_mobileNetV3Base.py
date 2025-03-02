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
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=weights)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                for param in m.parameters():
                    param.requires_grad = True

        in_features = self.model.classifier[3].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)
        nn.init.kaiming_normal_(self.model.classifier[-1].weight, nonlinearity='linear') #new 

    def forward(self, x):
        model_out = self.model(x)
        #out = torch.clamp(model_out, 0, 1)
        #out = torch.sigmoid(model_out)
        #out = (torch.tanh(model_out) + 1) / 2
        
        return model_out
    
def main():
    model = BallLocalization()
    dummy_input = torch.randn(1,3, 224, 224)  # Batch size of 64
    output = model(dummy_input)
    print(output.size())

if __name__ == "__main__":
    main()
