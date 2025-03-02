import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.binaryClassifier.FoosballDatasetRaw import FoosballDatasetRaw
# Define only ToTensor transformation (NO normalization or augmentation)
transform = transforms.Compose([
    transforms.ToTensor()  # Convert to tensor without normalization
])

def compute_mean_std():
    # Load dataset
    dataset = FoosballDatasetRaw(images_dir="data/images", json_path="data/labels/labels.json", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize accumulators
    mean = torch.zeros(3)  # RGB channels
    std = torch.zeros(3)
    num_samples = 0

    # Compute mean and std over dataset
    for i, batch in enumerate(dataloader):
        print(f"Batch: {i+1}")
        batch_samples = batch.size(0)  # Batch size
        num_samples += batch_samples

        mean += batch.mean(dim=[0, 2, 3]) * batch_samples
        std += batch.std(dim=[0, 2, 3]) * batch_samples
        #print(mean, std)
    mean /= num_samples
    std /= num_samples


    return mean, std
def main():
    # Final mean and std
    mean, std = compute_mean_std()


    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")

if __name__ == "__main__":
    main()