import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
from flask import Flask, request, jsonify,send_from_directory
import os
import io

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace = True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )

    def forward(self, x):
        new_features = self.net(x)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []

        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes = 41, growth_rate = 32, block_layers = [6, 12, 24, 16], init_channels = 64):
        super().__init__()
        
        self.growth_rate = growth_rate
        channels = init_channels

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, channels, growth_rate)
            self.blocks.append(block)
            channels += num_layers * growth_rate

            if i != len(block_layers) - 1:
                trans = TransitionLayer(channels, channels // 2)
                self.blocks.append(trans)
                channels = channels // 2

        self.final_bn = nn.BatchNorm2d(channels)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_bn(x)
        x = nn.functional.relu(x, inplace = True)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet(num_classes=6).to(device)

model.load_state_dict(torch.load("backend/model.pth", map_location=device))

model.eval()

classes = [
    "Atelectasis", "Effusion", "Infiltration", "Mass", "No Finding", "Nodule"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  
    transforms.Normalize((0.5,), (0.5,))  
])
def get_img(file):
    img = Image.open(io.BytesIO(file.read())).convert("L") 
    img = transform(img).unsqueeze(0).to(device)  
    return img

def get_probabilities(output, top=3):
    probs = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
    sorted_indices = probs.argsort()[::-1]

    # Check if top prediction is "No Finding"
    if classes[sorted_indices[0]] == "No Finding":
        return ["No Finding"], [f"{100.0 * probs[sorted_indices[0]]:.2f}%"]

    # Otherwise, exclude "No Finding" and return top N
    filtered = [(i, probs[i]) for i in sorted_indices if classes[i] != "No Finding"]
    top_filtered = filtered[:top]

    sorted_labels = [classes[i] for i, _ in top_filtered]
    sorted_probs = [f"{100.0 * p:.2f}%" for _, p in top_filtered]

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

    predicted_class = classes[predicted.item()]
    labs, probs = get_probabilities(output, 3)

    payload = {
        "prediction": predicted_class,
        "labels": labs,
        "probabilities": probs,
    }

    
    return jsonify(payload)



@app.route('/')
def serve_react():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

