import torch
from torchvision import models, transforms
from torch import nn
import requests
from PIL import Image
from io import BytesIO
import os
import argparse

# Tạo một parser
parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument('--image_url', type=str, help='URL of the image to classify')
args = parser.parse_args()
image_url = args.image_url

# Định nghĩa các phép biến đổi tiền xử lý
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6689, 0.6875, 0.7218], std=[0.3951, 0.3775, 0.3541]),  # Normalize with current dataset
])

# Tải mô hình đã được huấn luyện
model_path = 'best_model.pt'
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # Cập nhật số lớp dựa vào bộ dữ liệu của bạn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Hàm load và biến đổi hình ảnh từ URL hoặc đường dẫn tệp
def load_and_transform_image(image_path, transform):
    if os.path.isfile(image_path):
        image = Image.open(image_path).convert('RGB')
    else:
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    return transform(image)

# Hàm suy luận để dự đoán lớp của hình ảnh
def predict_image(image_url, model, transform):
    image = load_and_transform_image(image_url, transform)
    image = image.to(device)
    image = image.unsqueeze(0)  # Thêm một chiều batch_size
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    classes = ['checked', 'other', 'unchecked']
    predicted_class = classes[predicted[0]]
    return predicted_class

# Predict
predicted_class = predict_image(image_url, model, transform)
print(f'Predicted class: {predicted_class}')
