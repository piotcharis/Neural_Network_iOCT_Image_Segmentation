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
