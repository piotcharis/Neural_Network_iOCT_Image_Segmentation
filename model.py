import torch
from torchvision import transforms, datasets, models
import lightning as L
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LitSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=13)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)['out']
    
# Load the model
model = LitSegmentation()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()  # Set the model to evaluation mode
model.to(device)  # Send the model to GPU if available
    

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

image_path = "./data/test/images/000003_d19dd6c3-895f-4f90-bdc6-0006b85b0c7c.png"  # Replace with the path to your image
image_tensor = preprocess_image(image_path)
image_tensor = image_tensor.to(device)  # Move to GPU if available

with torch.no_grad():  # No need to compute gradients during inference
    output = model(image_tensor)
    predicted_mask = torch.argmax(output, dim=1)  # Get the predicted class for each pixel

# Convert the predicted mask back to PIL image or numpy array for visualization
predicted_mask = predicted_mask.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
predicted_mask_image = Image.fromarray(predicted_mask.astype('uint8'))  # Convert to PIL Image
predicted_mask_image.save("output_mask.png")  # or save it using predicted_mask_image.save("output_mask.png")

def mask_to_color(mask):
    # Define the color map (you can customize this)
    colormap = np.array([
        [0, 0, 0],  # Background
        [229, 4, 2],  # ILM
        [49, 141, 171],  # RNFL
        [138, 61, 199],  # GCL
        [154, 195, 239],  # IPL
        [245, 160, 56],  # INL
        [232, 146, 141],  # OPL
        [245, 237, 105],  # ONL
        [232, 206, 208],  # ELM
        [128, 161, 54],  # PR
        [32, 207, 255],  # RPE
        [232, 71, 72],  # BM
        [212, 182, 222],  # CC
        [196, 45, 4],  # CS
    ])

    # Ensure the mask values are within the range of the colormap
    mask = mask % len(colormap)
    color_mask = colormap[mask]

    return color_mask

# Convert the predicted mask to color
color_mask = mask_to_color(predicted_mask)
color_mask_image = Image.fromarray(color_mask.astype('uint8'))

# Load the original image
original_image = Image.open(image_path).convert("RGB")

# Ensure that the original image and mask are the same size
original_image = original_image.resize(color_mask_image.size)

# Blend the original image and the color mask
blended_image = Image.blend(original_image, color_mask_image, alpha=0.5)

# Display or save the blended image
blended_image.save("blended_image.png")
