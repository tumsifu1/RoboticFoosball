import torch 
import torch.nn as nn
import torchvision.models as models

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 1) #replace the last layer with a single neuron for a probability
    
    def forward(self, x):
        x = self.model(x)
        return x