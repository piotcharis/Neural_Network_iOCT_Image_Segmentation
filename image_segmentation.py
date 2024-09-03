import os
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = TensorBoardLogger('tb_logs', name='image_segmentation')

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=True,
    mode='min'
)

# Data loader
class ImageSegmentationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(os.path.join(self.root, 'images'))
        self.masks = os.listdir(os.path.join(self.root, 'masks'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, 'images', self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.root, 'masks', self.masks[idx])).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    
# Data augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Model
class ImageSegmentationModel(L.LightningModule):
    def __init__(self):
        super(ImageSegmentationModel, self).__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)['out']
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # Log the training loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        # Log the validation loss
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
# Train
def train() -> None:
    model = ImageSegmentationModel()
    model = model.to(device)
    dataset = ImageSegmentationDataset('data', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    trainer = L.Trainer(max_epochs=100, logger=logger, callbacks=[early_stop_callback])
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == '__main__':
    train()