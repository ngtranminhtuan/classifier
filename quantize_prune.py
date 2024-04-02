import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

# Define settings
data_dir = 'checkbox_state_v2/data'
model_path = 'best_model.pt'
pruned_and_finetuned_model_path = 'pruned_finetuned_model.pt'
quantized_model_path = 'quantized_model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Update based on your dataset
batch_size = 32
learning_rate = 0.001
momentum = 0.9
num_epochs_finetune = 500

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6689, 0.6875, 0.7218], std=[0.3951, 0.3775, 0.3541]),
])

# Load datasets
train_dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
val_dataset = ImageFolder(root=f'{data_dir}/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the best model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Apply pruning to the model
parameters_to_prune = (
    (model.layer1[0].conv1, 'weight'),
    (model.layer1[0].conv2, 'weight'),
    (model.layer2[0].conv1, 'weight'),
    (model.layer2[0].conv2, 'weight'),
)
for module, name in parameters_to_prune:
    prune.l1_unstructured(module, name=name, amount=0.2)
    prune.remove(module, name=name)  # Make pruning permanent

# Fine-tune the pruned model
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs_finetune):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs_finetune}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), pruned_and_finetuned_model_path)
print("Pruned and fine-tuned model saved.")

# Apply dynamic quantization to the fine-tuned, pruned model
model.eval()  # Make sure the model is in eval mode before quantization
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

torch.save(quantized_model.state_dict(), quantized_model_path)
print("Dynamically quantized model saved.")
