import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models
class BallLocalization(nn.Module):
    def __init__(self):
        super(BallLocalization, self).__init__()

        self.model = models.mobilenet_v3_large(pretrained=True)

        in_features = self.model.classifier[3].in_features

        self.model.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)
    
def main():
    model = BallLocalization()
    dummy_input = torch.randn(1,3, 2304, 1296)  # Batch size of 64
    output = model(dummy_input)
    print(output.size())

if __name__ == "__main__":
    main()
