import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_mean_std(data_dir, image_size=224, batch_size=256, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    print("Computing mean and std...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)  # (B, C, H*W)

        mean += images.mean(2).sum(0)  # sum over batch
        std += images.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    return mean, std

if __name__ == "__main__":
    data_path = "/ssd/hbmun/supcon/etri"
    mean, std = compute_mean_std(data_path)
    print("Mean:", [round(m, 3) for m in mean.tolist()])
    print("Std: ", [round(s, 3) for s in std.tolist()])
