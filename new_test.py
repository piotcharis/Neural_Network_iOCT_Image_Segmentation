import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, Resize
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('tb_logs', name='image_segmentation')

# Custom Dataset for grayscale images and masks
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])  # assuming mask has the same name as the image
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = np.expand_dims(image, axis=-1)  # add channel dimension
        mask = np.expand_dims(mask, axis=-1)  # add channel dimension
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = torch.squeeze(mask, dim=-1)  # Remove the channel dimension from mask
        
        return image, mask

# UNet Model Definition
class UNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(1, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))
        e5 = self.encoder5(self.maxpool(e4))
        
        # Decoder
        d4 = self.upconv4(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.final_conv(d1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, masks.squeeze(1).long())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, masks.squeeze(1).long())
        self.log('val_loss', loss)
        return loss

# Augmentations and transforms
def get_transform():
    return Compose([
        Resize(512, 512),  # Resize all images and masks to 512x512
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

# Data Loaders
def create_dataloaders(train_images, train_masks, val_images, val_masks, batch_size=8):
    train_dataset = SegmentationDataset(train_images, train_masks, transform=get_transform())
    val_dataset = SegmentationDataset(val_images, val_masks, transform=get_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

# Training
def train_model(train_images, train_masks, val_images, val_masks, num_classes, max_epochs=20):
    train_loader, val_loader = create_dataloaders(train_images, train_masks, val_images, val_masks)
    
    model = UNet(num_classes)
    
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

# Example usage
if __name__ == "__main__":
    train_images = "data/train/images"
    train_masks = "data/train/masks"
    val_images = "data/val/images"
    val_masks = "data/val/masks"
    
    num_classes = 14  # For example, background + 2 classes
    train_model(train_images, train_masks, val_images, val_masks, num_classes)
