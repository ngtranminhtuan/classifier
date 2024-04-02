import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Giả sử 'your_dataset' là tập dữ liệu của bạn
dataset = datasets.ImageFolder(
    root='checkbox_state_v2/data',
    transform=transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
)

# Loader để duyệt qua tập dữ liệu
loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)

# Hàm để tính mean và std
def calculate_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

mean, std = calculate_mean_std(loader)
print(f'Mean: {mean}')
print(f'Std: {std}')
