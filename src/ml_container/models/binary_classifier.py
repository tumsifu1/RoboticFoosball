import torch
import torch.nn as nn
import torchvision.models as models

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        
        # Use a smaller, faster model instead of ResNet18
        self.model = models.mobilenet_v3_small(pretrained=False)

        # Adjust the final classification layer
        feature_size = self.model.classifier[0].in_features
        self.model.classifier = nn.Linear(feature_size, 1)
    
    def forward(self, x):
        return self.model(x)