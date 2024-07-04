from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import io
import os

app = Flask(__name__)

# Define the ResNet18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
model.load_state_dict(torch.load('cifar_net.pth', map_location=device))
model.eval()

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    # Preprocess the image
    img = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img)
    
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    
    return jsonify({'prediction': classes[predicted.item()]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)