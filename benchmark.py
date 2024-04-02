import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import time

# Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'checkbox_state_v2/data/val'
batch_size = 32
model_path = 'best_model.pt'

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load Model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Benchmark Function
def benchmark_model(model, dataloader):
    model.eval()
    start_time = time.time()
    total = 0
    correct = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = (end_time - start_time) / total

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Average Inference Time: {inference_time:.6f} seconds per sample')

# Run Benchmark
if __name__ == '__main__':
    benchmark_model(model, dataloader)
