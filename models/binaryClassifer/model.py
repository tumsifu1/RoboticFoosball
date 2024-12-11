import torch
import torch.nn as nn
import torchvision.models as models
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Get the size of the features from the penultimate layer
        feature_size = self.model.fc.in_features

        # Replace the FC layer with a new one for binary classification
        self.model.fc = nn.Linear(feature_size, 1)
    
    def forward(self, x):
        return self.model(x)
