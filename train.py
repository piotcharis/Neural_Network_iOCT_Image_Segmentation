import torch
from torchvision import transforms, datasets, models
import pytorch_lightning as L
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = TensorBoardLogger("tb_logs", name="segmentation_model")

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Number of epochs to wait for improvement
    mode='min'  # Mode is 'min' for loss (lower is better)
)

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

        image = Image.open(img_path).convert("L")  # Keep it as grayscale
        image = image.convert("RGB")  # Convert it to RGB by replicating the single channel
        mask = Image.open(mask_path).convert("L")
        
        # Mask has values 0-12; we need to convert it to 0-255
        mask = np.array(mask)
        mask = (mask / 12 * 255).astype(np.uint8)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

class LitSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=13)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch
        images, targets = images.to(self.device), targets.to(self.device)  # Move to GPU
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("val_loss", loss)
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
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),  # Resize to match input size
            transforms.ToTensor()  # Convert to tensor; keeps values as is
        ])

        train_images_dir = os.path.join(self.data_dir, "train/images")
        train_masks_dir = os.path.join(self.data_dir, "train/masks")
        val_images_dir = os.path.join(self.data_dir, "val/images")
        val_masks_dir = os.path.join(self.data_dir, "val/masks")

        self.train_dataset = CustomSegmentationDataset(
            images_dir=train_images_dir,
            masks_dir=train_masks_dir,
            transform=transform,
            target_transform=target_transform
        )

        self.val_dataset = CustomSegmentationDataset(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            transform=transform,
            target_transform=target_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


if __name__ == "__main__":
    model = LitSegmentation()
    data = SegmentationData(data_dir="./data")  # Ensure this points to your data folder
        
    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=30,  # Upper limit
        accelerator="gpu", 
        devices=1,
        logger=logger,
        callbacks=[early_stopping]
    )
    
    trainer.fit(model, data)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")