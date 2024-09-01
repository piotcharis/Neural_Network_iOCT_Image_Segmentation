import torch
from torchvision import transforms, models
import lightning as L
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class CustomSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))

        # Define a color to class mapping (example)
        self.color_to_class = {
            (0, 0, 0): 0,  # Background
            (229, 4, 2): 1,  # ILM
            (49, 141, 171): 2,  # RNFL
            (138, 61, 199): 3,  # GCL
            (154, 195, 239): 4,  # IPL
            (245, 160, 56): 5,  # INL
            (232, 146, 141): 6,  # OPL
            (245, 237, 105): 7,  # ONL
            (232, 206, 208): 8,  # ELM
            (128, 161, 54): 9,  # PR
            (32, 207, 255): 10,  # RPE
            (232, 71, 72): 11,  # BM
            (212, 182, 222): 12,  # CC
            (196, 45, 4): 13,  # CS
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # Load mask as RGB

        if self.transform:
            image = self.transform(image)

        # Convert RGB mask to class indices
        mask = self.rgb_to_class(mask)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def rgb_to_class(self, mask):
        # Convert the RGB mask to a single channel of class indices
        mask = torch.from_numpy(np.array(mask))
        class_mask = torch.zeros((mask.size(0), mask.size(1)), dtype=torch.long)

        for color, class_id in self.color_to_class.items():
            class_mask[(mask == torch.tensor(color)).all(dim=-1)] = class_id

        return class_mask


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
    trained_model = LitSegmentation()
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()  # Set the model to evaluation mode
    return trained_model


def predict(trained_model, image_path):
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
        output = trained_model(image)['out']

    # Get the predicted mask (assuming you want the class with the highest score)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0)

    return predicted_mask


def class_to_rgb(class_mask, class_to_color):
    """Convert a class mask to an RGB image using the class-to-color mapping."""
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_to_color.items():
        rgb_mask[class_mask == class_id] = color

    return Image.fromarray(rgb_mask)


if __name__ == "__main__":
    # Load the trained model
    model = load_model("model.pth")

    # Specify the path to a new image
    image_path = "./data/test/images/000003_d19dd6c3-895f-4f90-bdc6-0006b85b0c7c.png"

    # Make a prediction
    predicted_mask = predict(model, image_path)

    # Convert the predicted mask back to RGB
    class_to_color = {v: k for k, v in model.color_to_class.items()}  # Reverse the mapping
    predicted_rgb_mask = class_to_rgb(predicted_mask.numpy(), class_to_color)

    # Save or display the predicted mask
    predicted_rgb_mask.save("predicted_rgb_mask.png")
    predicted_rgb_mask.show()
