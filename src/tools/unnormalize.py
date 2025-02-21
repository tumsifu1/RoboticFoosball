import torch

def unnormalize(image):
    mean = [0.1249, 0.1399, 0.1198]
    std = [0.1205, 0.1251, 0.1123]
    """Undo normalization for visualization."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean #tensor-wise math 