import torch
from torchvision import transforms, datasets, models
import lightning as L
from PIL import Image
import os
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are in grayscale

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

class LitSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=21)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
def load_model(model_path):
    model = LitSegmentation()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, image_path):
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    # Run the model
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)['out']

    # Get the predicted mask (assuming you want the class with the highest score)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0)

    return predicted_mask

if __name__ == "__main__":
    # Load the trained model
    model = load_model("model.pth")

    # Specify the path to a new image
    image_path = "./data/test/images/000003_d19dd6c3-895f-4f90-bdc6-0006b85b0c7c.png"

    # Make a prediction
    predicted_mask = predict(model, image_path)

    # Convert the predicted mask to a PIL image (optional)
    predicted_mask_pil = transforms.ToPILImage()(predicted_mask.byte())

    # Save or display the predicted mask
    predicted_mask_pil.save("predicted_mask.png")
    predicted_mask_pil.show()
