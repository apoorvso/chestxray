import torch
from torchvision import transforms,models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
from flask import Flask, request, jsonify,url_for

import os
import io
import requests
import torch.optim as optim
from math import sqrt
import uuid

# === Together.ai API settings ===
API_KEY = "e3e9a35c04c7f6c6a476f4f8ee8ad130ac66b43c1dd06d701a81d5fa8ed6b27e"
API_URL = "https://api.together.xyz/v1/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
# View classificatio 

class CNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),  # now 1→16
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
    
# === DenseNet Model Definition ===
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        new_features = self.net(x)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = [DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes=6, growth_rate=32, block_layers=[6, 12, 24, 16], init_channels=64):
        super().__init__()
        channels = init_channels
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MultiOutputModel(nn.Module):
    def __init__(self): # Corrected the typo: _init_ to __init__
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)  # Define the backbone using a pretrained ResNet18
        
        # Modify the first convolutional layer to accept 1 input channel (grayscale)
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #This line can be removed as we are replacing conv1 above

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head    = nn.Linear(in_features, 1)  # single continuous output
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        return self.age_head(x).squeeze(1), self.gender_head(x)

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input  = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.residual_layer = nn.Sequential(*[Conv_ReLU_Block() for _ in range(18)])
        self.output = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.relu   = nn.ReLU(inplace=True)

        # Kaiming initialization (same as original)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        return out + residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

age_model = MultiOutputModel().to(device)
# Load the state dictionary with strict=False to ignore unexpected keys
age_model.load_state_dict(torch.load("backend/age_gender.pth", map_location=device), strict=False) 
age_model.eval()


model = DenseNet(num_classes=6).to(device)
model.load_state_dict(torch.load("backend/model.pth", map_location=device))
model.eval()

view_model = CNN(in_channels=1).to(device)

# load raw state‑dict
view_ckpt = torch.load("backend/view_position_classification_model.pth",
                       map_location="cpu")

# patch conv1 weights: average the RGB → gray
# key is 'features.0.weight' because that's the first Conv2d
w_rgb = view_ckpt["features.0.weight"]                # shape [16,3,3,3]
view_ckpt["features.0.weight"] = w_rgb.mean(dim=1, 
                                            keepdim=True)  # [16,1,3,3]

# load patched weights
view_model.load_state_dict(view_ckpt, strict=True)
view_model.eval()


vdsr_state   = torch.load("backend/model_weights_vdsr_finetuned.pth", map_location=device, weights_only=False)
         # pull state‑dict out of the saved model object

model_vdsr  = Net().to(device)
model_vdsr.load_state_dict(vdsr_state, strict=True)    # should load with zero missing / unexpected keys
model_vdsr.eval()

view_labels = {0: "AP", 1: "PA"}
classes = ["Atelectasis", "Effusion", "Infiltration", "Mass", "No Finding", "Nodule"]

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
    probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
    sorted_indices = probs.argsort()[::-1]
    if classes[sorted_indices[0]] == "No Finding":
        return ["No Finding"], [f"{100.0 * probs[sorted_indices[0]]:.2f}%"]
    filtered = [(i, probs[i]) for i in sorted_indices if classes[i] != "No Finding"]
    top_filtered = filtered[:top]
    sorted_labels = [classes[i] for i, _ in top_filtered]
    sorted_probs = [f"{100.0 * p:.2f}%" for _, p in top_filtered]
    return sorted_labels, sorted_probs

# === Report Generator using Together.ai ===
def generate_radiology_report(age, gender, view, findings):
    prompt = f"""<|system|>
You are a helpful radiologist assistant.
<|user|>
Generate a chest X-ray report for a {age}-year-old {gender}, {view} view, with the following findings: {', '.join(findings)}.
First, define each finding in 1–2 sentences. Then, generate the radiology report.
<|assistant|>
"""

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["<|user|>", "<|system|>"]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    else:
        raise Exception(f"Together API Error {response.status_code}: {response.text}")

# === Flask App ===
base_dir   = os.path.dirname(os.path.dirname(__file__))  
static_dir = os.path.join(base_dir, "static")

app = Flask(
    __name__,
    static_folder=static_dir,    # now Flask will serve CWD/../static
    static_url_path="/static"    # at the /static URL prefix
)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "✅ Backend is running with Together.ai report generation"

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # ─── Disease classification ───
    image = get_img(file)  # uses your existing get_img for DenseNet
    with torch.no_grad():
        disease_output = model(image)
        _, predicted = torch.max(disease_output, 1)
        labs, probs = get_probabilities(disease_output, 3)
        predicted_class = classes[predicted.item()]

    # ─── Age & Gender prediction ───
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                       # [1,H,W]
        transforms.Normalize([0.485], [0.229])
    ])
    file.seek(0)
    image_gray = transform_gray(
        Image.open(io.BytesIO(file.read())).convert("L")
    ).unsqueeze(0).to(device)                       # [1,1,224,224]

    with torch.no_grad():
        age_pred, gender_logits = age_model(image_gray)
    age_pred = round(age_pred.item(), 1)
    gender_idx = torch.argmax(F.softmax(gender_logits, dim=1)).item()
    gender_label = "Male" if gender_idx == 0 else "Female"

    # ─── View‐Position prediction ───
    transform_view = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                       # [1,H,W]
        transforms.Normalize([0.485], [0.229])
    ])
    file.seek(0)
    image_view = transform_view(
        Image.open(io.BytesIO(file.read())).convert("L")
    ).unsqueeze(0).to(device)                       # [1,1,224,224]

    with torch.no_grad():
        view_logits = view_model(image_view)
    view_idx   = torch.argmax(F.softmax(view_logits, dim=1), dim=1).item()
    view_label = "AP" if view_idx == 0 else "PA"

    # ─── VDSR enhancement ───
    # read and preprocess grayscale
    file.seek(0)
    orig = Image.open(io.BytesIO(file.read())).convert("L")
    sr_tf = transforms.Compose([transforms.ToTensor()])
    inp_sr = sr_tf(orig).unsqueeze(0).to(device)       # [1,1,H,W]
    with torch.no_grad():
        out_sr = model_vdsr(inp_sr)                    # [1,1,H,W]

    # convert to PIL
    arr = (out_sr.squeeze().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    pil_sr = Image.fromarray(arr)

    # save into Flask static folder
    enhanced_dir = os.path.join(app.static_folder, "enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(enhanced_dir, fname)
    pil_sr.save(save_path)

    # build a URL that Flask will serve
    enhanced_url = url_for("static", filename=f"enhanced/{fname}", _external=True)

    return jsonify({
        "prediction":     predicted_class,
        "labels":         labs,
        "probabilities":  probs,
        "age":            age_pred,
        "gender":         gender_label,
        "view":           view_label,
        "enhanced_url":   enhanced_url
    })


@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json()
    if not all(k in data for k in ["age", "gender", "view", "findings"]):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        report = generate_radiology_report(data["age"], data["gender"], data["view"], data["findings"])
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
