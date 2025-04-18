import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dataset_path = dataset_path = "/Users/bhavya.motukuri.11/Desktop/ALZHEIMER/data_split"  # Adjusted path
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)


model_path = os.path.join(os.path.dirname(__file__), 'alzheimers_model.pth')  # Model file path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
model.load_state_dict(torch.load(model_path))
model = model.to(device)


model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {test_accuracy:.2f}%")


class_names = test_dataset.classes
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\nClass-wise Accuracy:")

for i, class_name in enumerate(class_names):
    class_correct = conf_matrix[i, i]
    class_total = conf_matrix[i].sum()
    class_accuracy = 100 * class_correct / class_total if class_total > 0 else 0
    print(f"{class_name}: {class_accuracy:.2f}%")

