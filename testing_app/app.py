import os
import cv2
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '../model')
from unet import UNet

app = Flask(__name__)
CORS(app)

# Bind to PORT if defined, otherwise default to 5000
port = int(os.environ.get('PORT', 5000))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize default model
default_model_path = 'model.pth'
model = UNet(n_classes=14, img_channels=1).to(device)
model.load_state_dict(torch.load(default_model_path, map_location=device, weights_only=True))
model.eval()

# Path to store the uploaded model
uploaded_model_path = 'uploaded_model.pth'

# Preprocess the image before passing it to the model
def preprocess_image(image):
    img = image.convert('L')
    img = np.array(img)
    img = cv2.resize(img, (256, 512))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.tensor(img, dtype=torch.float32).to(device)
    return img_tensor

color_map = np.array([
    [0, 0, 0], [229, 4, 2], [49, 141, 171], [138, 61, 199], [154, 195, 239],
    [245, 160, 56], [232, 146, 141], [245, 237, 105], [232, 206, 208],
    [128, 161, 54], [32, 207, 255], [232, 71, 72], [212, 182, 222], [196, 45, 4]
])

def apply_colormap(mask, color_map):
    return color_map[mask]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file)

    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
        
    predicted_mask_color = apply_colormap(predicted_mask, color_map)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    
    # Resize the image to 512x256
    image = cv2.resize(np.array(image), (256, 512))
    
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.imshow(predicted_mask_color, alpha=0.5)
    plt.title('Predicted Mask')
    
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error": "No model file uploaded"}), 400

    file = request.files['model']
    file.save(uploaded_model_path)

    # Load the new model
    global model
    model = UNet(n_classes=14, img_channels=1).to(device)
    model.load_state_dict(torch.load(uploaded_model_path, map_location=device, weights_only=True))
    model.eval()

    return jsonify({"message": "Model uploaded successfully"}), 200

@app.route('/revert_model', methods=['POST'])
def revert_model():
    global model
    model = UNet(n_classes=14, img_channels=1).to(device)
    model.load_state_dict(torch.load(default_model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Delete the uploaded model
    if os.path.exists(uploaded_model_path):
        os.remove(uploaded_model_path)

    return jsonify({"message": "Reverted to default model"}), 200

if __name__ == '__main__':
    app.run(debug=True)
