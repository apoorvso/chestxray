import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
from flask import Flask, request, jsonify
import os
import io

class ImageClassificationNetwork(nn.Module):
    def __init__(self):
        super(ImageClassificationNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # (16 x 16 x 16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.AvgPool2d(2, 2)  # (32 x 8 x 8)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.AvgPool2d(2, 2)  # (64 x 4 x 4)
        
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.AvgPool2d(2, 2)  # (128 x 2 x 2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool5 = nn.AvgPool2d(2, 2)  # (256 x 1 x 1)

        self.fc = nn.Linear(256 * 1 * 1, 10)  # Output for 10 classes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        
        x = x.view(-1, 256 * 1 * 1)  # Flatten
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassificationNetwork().to(device)

model.load_state_dict(torch.load("backend/model.pth", map_location=device))

model.eval()

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  
    transforms.Normalize((0.5,), (0.5,))  
])
def get_img(file):
    img = Image.open(io.BytesIO(file.read())).convert("RGB") 
    img = transform(img).unsqueeze(0).to(device)  
    return img

def get_probabilities(output, top=3):
    probs = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
    probs = probs.squeeze().cpu().numpy()
    sorted_indices = probs.argsort()[::-1]
    sorted_probs = probs[sorted_indices][:top]
    sorted_probs = [f"{100.0 * p:.2f}%" for p in sorted_probs]
    sorted_labels = [cifar10_classes[i] for i in sorted_indices][:top]

    return sorted_labels, sorted_probs

app = Flask(__name__)
CORS(app)  # Need CORS in order to allow for Cross Origin Requests.


@app.route('/', methods=['GET'])
def home():
    return "This is the home page for backend server"

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = get_img(file)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = cifar10_classes[predicted.item()]
    labs, probs = get_probabilities(output, 3)

    payload = {
        "prediction": predicted_class,
        "labels": labs,
        "probabilities": probs,
    }
    return jsonify(payload)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

