import torch
from PIL import Image
import pytorch_lightning as L
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt

class LitSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=13)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)['out']

model = LitSegmentation()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "./data/test/images/000003_d19dd6c3-895f-4f90-bdc6-0006b85b0c7c.png"  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_image)

predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, cmap="gray")
plt.show()

print(np.unique(predicted_mask))
