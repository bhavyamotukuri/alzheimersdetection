import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image


image_path = 'nondemtest.jpg' 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)


model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("alzheimers_model.pth", map_location=device))
model = model.to(device)
model.eval()


with torch.no_grad():
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)


test_dataset = datasets.ImageFolder('data_split/test', transform=transform)
classes = test_dataset.classes

print(f" Predicted class: {classes[predicted_class.item()]}")
