import torch
from torchvision import transforms, datasets, models
import lightning as L
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
            (229, 4, 2): 1,  # Class 1
            (49, 141, 171): 2,  # Class 2
            (138, 61, 199): 3,  # Class 3
            (154, 195, 239): 4,  # Class 4
            (0, 0, 0): 0,
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


class SegmentationData(L.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

        # Specify the paths for images and masks
        train_images_dir = os.path.join(self.data_dir, "train/images")
        train_masks_dir = os.path.join(self.data_dir, "train/masks")

        # Create the custom dataset
        self.train_dataset = CustomSegmentationDataset(
            images_dir=train_images_dir,
            masks_dir=train_masks_dir,
            transform=transform,
            target_transform=target_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)


if __name__ == "__main__":
    model = LitSegmentation()
    data = SegmentationData(data_dir="./data")  # Ensure this points to your data folder
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, data)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
