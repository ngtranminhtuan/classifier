import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Preprocessing and Augmentation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6689, 0.6875, 0.7218], std=[0.3951, 0.3775, 0.3541]),  # Normalize with current dataset
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root='checkbox_state_v2/data/train', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root='checkbox_state_v2/data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model and change output shape
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # Replace last output
model_path = 'best_model.pt'
model.load_state_dict(torch.load(model_path))

# Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.85)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Changed .view() to .reshape()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Khởi tạo danh sách để theo dõi giá trị qua các lần đánh giá
epoch_losses = []
epoch_acc1s = []
epoch_acc5s = []

def evaluate_and_plot(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    topk_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    topk = min(3, len(train_dataset.classes))  # Ensure topk is not greater than number of classes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += labels.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            acc1, acctopk = accuracy(outputs, labels, topk=(1, topk))
            top1_correct += acc1[0].item() * labels.size(0) / 100  # Convert percentages back to numbers
            topk_correct += acctopk[0].item() * labels.size(0) / 100  # Convert percentages back to numbers

    # Calculate the confusion matrix
    confusion_mtx = confusion_matrix(all_labels, all_preds)
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_plot/confusion_matrix_epoch_{epoch}.png')
    plt.close()

    # Now calculate average loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_top1_acc = (top1_correct / total_samples) * 100  # Convert back to percentage
    epoch_topk_acc = (topk_correct / total_samples) * 100  # Convert back to percentage

    # Update lists for plotting
    epoch_losses.append(epoch_loss)
    epoch_acc1s.append(epoch_top1_acc)
    epoch_acc5s.append(epoch_topk_acc)

    # Plot Loss
    axs[0].plot(epoch_losses, 'bo-')
    axs[0].set_title('Train Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    # Plot Top-1 Accuracy
    axs[1].plot(epoch_acc1s, 'bo-')
    axs[1].set_title(f'Top-1 Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    # Plot Top-k Accuracy
    axs[2].plot(epoch_acc5s, 'bo-')
    axs[2].set_title(f'Top-{topk} Accuracy')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.close()

    print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Top-1 Acc: {epoch_top1_acc:.2f}%, Top-{topk} Acc: {epoch_topk_acc:.2f}%')

    return epoch_loss, epoch_top1_acc, epoch_topk_acc

# Cập nhật vòng lặp huấn luyện để bao gồm đánh giá
best_val_loss = 0.3106
best_val_acc1 = 91.35
patience = 200  # Number of epochs to wait for improvement before stopping
patience_counter = 0  # Counter for tracking the number of epochs without improvement

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    global best_val_loss, best_val_acc1, patience_counter

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Đánh giá sau mỗi epoch
        val_loss, val_acc1, _ = evaluate_and_plot(model, val_loader, epoch + 1)
        
        # Check for improvement
        if val_loss < best_val_loss and val_acc1 >= best_val_acc1:
            best_val_loss = val_loss
            best_val_acc1 = val_acc1
            patience_counter = 0  # Reset patience_counter
            print(f"New best model found! Epoch: {epoch+1}, Loss: {val_loss:.4f}, Top-1 Acc: {val_acc1:.2f}%")
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered. Stopping training at epoch {epoch+1}.')
            break

    print("Training complete.")
    return model

# Huấn luyện và đánh giá mô hình
trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=500)

