import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from sklearn.metrics import jaccard_score
from unet import UNet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = TensorBoardLogger('tb_logs', name='image_segmentation')

class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)  # Shape: [height, width, channels]
        mask = torch.tensor(self.masks[idx], dtype=torch.long)  # Shape: [height, width]

        # Add channel dimension and reorder dimensions for PyTorch: [channels, height, width]
        image = image.permute(2, 0, 1)  # Permute to [channels, height, width]
        return image, mask
    
# Load images and masks
def load_data():
    SIZE_X = 256
    SIZE_Y = 256
    TRAIN_PATH_X = 'data/train/images'
    TRAIN_PATH_Y = 'data/train/masks'

    train_images = []
    train_masks = []

    # Load images
    for img_path in glob.glob(os.path.join(TRAIN_PATH_X, "*.png")):
        img = cv2.imread(img_path, 0)  # Load as grayscale
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)

    # Load masks
    for mask_path in glob.glob(os.path.join(TRAIN_PATH_Y, "*.png")):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        train_masks.append(mask)

    # Convert lists to numpy arrays
    train_images = np.array(train_images)  # Shape: [num_samples, height, width]
    train_masks = np.array(train_masks)    # Shape: [num_samples, height, width]

    # Add channel dimension
    train_images = np.expand_dims(train_images, axis=3)  # Shape: [num_samples, height, width, 1]
    train_masks = np.expand_dims(train_masks, axis=3)    # Shape: [num_samples, height, width, 1]

    # Encode masks if necessary
    labelencoder = LabelEncoder()
    n_classes = len(np.unique(train_masks))
    train_masks_reshaped = train_masks.reshape(-1)  # Flatten
    train_masks_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks = train_masks_encoded.reshape(*train_masks.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.10, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Load the data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

# Create DataLoader
train_dataset = SegmentationDataset(X_train, y_train)
val_dataset = SegmentationDataset(X_val, y_val)
test_dataset = SegmentationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Instantiate the model
model = UNet(in_channels=1, out_channels=14)
model.to(device)

# Set up the trainer
trainer = Trainer(max_epochs=50, logger=logger)

# Fit the model
trainer.fit(model, train_loader, val_loader)

# Evaluate on test data
def calculate_iou(y_true, y_pred, n_classes):
    iou_scores = []
    for c in range(n_classes):
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        iou = jaccard_score(y_true_c.flatten(), y_pred_c.flatten())
        iou_scores.append(iou)
    return np.mean(iou_scores)

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in test_loader:
        y_pred = model(x).argmax(dim=1)
        all_preds.append(y_pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())

# Flatten the lists of arrays
y_pred_all = np.concatenate(all_preds, axis=0)
y_true_all = np.concatenate(all_labels, axis=0)

# Calculate Mean IoU
iou = calculate_iou(y_true_all.flatten(), y_pred_all.flatten(), n_classes=14)
print("Mean IoU =", iou)

# Save model
torch.save(model.state_dict(), 'pytorch_model.pth')

# Plot loss and accuracy (assuming history is available)
# History attributes like 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy' should be accessed from logger

# Example for plotting loss
import matplotlib.pyplot as plt

# Assume trainer logger or any logging mechanism provides access to history
loss = trainer.logger.experiment.history['train_loss']
val_loss = trainer.logger.experiment.history['val_loss']
epochs = range(1, len(loss) + 1)

# Plot and save loss
plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
