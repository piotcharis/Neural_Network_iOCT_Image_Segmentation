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
        
        # Convert mask to numpy array and ensure it is long tensor with discrete class values
        mask = transforms.Resize((256, 256))(mask)
        mask = np.array(mask, dtype=np.int64)  # Ensure mask is not normalized and is integer type
        mask = torch.tensor(mask, dtype=torch.long)
                
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
        self.model.classifier[4] = torch.nn.Conv2d(512, 14, kernel_size=(1, 1), stride=(1, 1))
        self.loss = torch.nn.CrossEntropyLoss()

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

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

# Visualize the predicted mask on a sample image from the dataset
def visualize_prediction() -> None:
    model = ImageSegmentationModel()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    model.to(device)
    dataset = ImageSegmentationDataset('data', transform=transform)
    image, mask = dataset[0]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    
    # Visualize the image and mask
    image = transforms.ToPILImage()(image.squeeze())
    mask = Image.fromarray(predicted_mask.astype('uint8'))
    mask.putpalette([
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
    # Save the images
    image.save('sample_image.png')
    mask.save('predicted_mask.png')
    
if __name__ == '__main__':
    train()
    visualize_prediction()