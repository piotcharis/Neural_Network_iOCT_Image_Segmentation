import os
import sys
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.classification import MulticlassJaccardIndex
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn.functional as F
from PIL import Image


# Set print options to display full arrays
np.set_printoptions(threshold=sys.maxsize)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
SIZE_X = 512
SIZE_Y = 256
n_classes = 14  # Number of classes for segmentation

# Paths
TRAIN_PATH_X = './data/train/images'
TRAIN_PATH_Y = './data/train/masks'
TEST_PATH_X = './data/test/images'
TEST_PATH_Y = './data/test/masks'

# Get image file paths
train_ids_x = next(os.walk(TRAIN_PATH_X))[2]
train_ids_y = next(os.walk(TRAIN_PATH_Y))[2]
test_ids_x = next(os.walk(TEST_PATH_X))[2]
test_ids_y = next(os.walk(TEST_PATH_Y))[2]

# Capture and resize training images
train_images = []
for img_path in glob.glob(os.path.join(TRAIN_PATH_X, "*.png")):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_images.append(img)

train_images = np.array(train_images)

# Capture and resize masks
train_masks = []
for mask_path in glob.glob(os.path.join(TRAIN_PATH_Y, "*.png")):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
    train_masks.append(mask)

train_masks = np.array(train_masks)

# Capture and resize test images
test_images = []
for img_path in glob.glob(os.path.join(TEST_PATH_X, "*.png")):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    test_images.append(img)
    
# Capture and resize test masks
test_masks = []
for mask_path in glob.glob(os.path.join(TEST_PATH_Y, "*.png")):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
    test_masks.append(mask)

test_masks = np.array(test_masks)

# Encode labels
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

# Encode test labels
n, h, w = test_masks.shape
test_masks_reshaped = test_masks.reshape(-1, 1)
test_masks_reshaped_encoded = labelencoder.fit_transform(test_masks_reshaped)
test_masks_encoded_original_shape = test_masks_reshaped_encoded.reshape(n, h, w)


# Expand dimensions and normalize images
train_images = np.expand_dims(train_images, axis=3)
train_images = train_images / 255.0  # Normalization

# Expand dimensions and normalize test images
test_images = np.expand_dims(test_images, axis=3)
test_images = test_images / 255.0  # Normalization

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
test_masks_input = np.expand_dims(test_masks_encoded_original_shape, axis=3)

# Train-test split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size=0.2, random_state=0)

print("Class values in the dataset are ... ", np.unique(y_train))

# Convert labels to one-hot encoded format
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)].reshape((*y.shape[:-1], num_classes))

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (N, 1, H, W)
y_train_cat = torch.tensor(y_train_cat, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (N, n_classes, H, W)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (N, 1, H, W)
y_test_cat = torch.tensor(y_test_cat, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (N, n_classes, H, W)

test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (N, 1, H, W)

# Define the U-Net model (or replace with your custom model)
class UNet(pl.LightningModule):
    def __init__(self, n_classes, img_height, img_width, img_channels):
        super(UNet, self).__init__()

        # Contraction path
        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(0.2)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout(0.3)

        # Expansive path
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout6 = nn.Dropout(0.2)

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout7 = nn.Dropout(0.2)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout8 = nn.Dropout(0.1)

        self.up9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dropout9 = nn.Dropout(0.1)

        self.conv10 = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        # Contraction path
        c1 = F.relu(self.conv1(x))
        c1 = F.relu(self.conv1_2(c1))
        c1 = self.dropout1(c1)
        p1 = self.pool1(c1)

        c2 = F.relu(self.conv2(p1))
        c2 = F.relu(self.conv2_2(c2))
        c2 = self.dropout2(c2)
        p2 = self.pool2(c2)

        c3 = F.relu(self.conv3(p2))
        c3 = F.relu(self.conv3_2(c3))
        c3 = self.dropout3(c3)
        p3 = self.pool3(c3)

        c4 = F.relu(self.conv4(p3))
        c4 = F.relu(self.conv4_2(c4))
        c4 = self.dropout4(c4)
        p4 = self.pool4(c4)

        c5 = F.relu(self.conv5(p4))
        c5 = F.relu(self.conv5_2(c5))
        c5 = self.dropout5(c5)

        # Expansive path
        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = F.relu(self.conv6(u6))
        c6 = F.relu(self.conv6_2(c6))
        c6 = self.dropout6(c6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = F.relu(self.conv7(u7))
        c7 = F.relu(self.conv7_2(c7))
        c7 = self.dropout7(c7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = F.relu(self.conv8(u8))
        c8 = F.relu(self.conv8_2(c8))
        c8 = self.dropout8(c8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = F.relu(self.conv9(u9))
        c9 = F.relu(self.conv9_2(c9))
        c9 = self.dropout9(c9)

        outputs = self.conv10(c9)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.squeeze(-1)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        y = y.squeeze(-1)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = UNet(n_classes=n_classes, img_height=SIZE_Y, img_width=SIZE_X, img_channels=1).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 150
batch_size = 16

train_loader = DataLoader(list(zip(X_train, y_train_cat)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test_cat)), batch_size=batch_size, shuffle=False)

train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            val_loss += loss.item() * inputs.size(0)
    
    val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
# Save the model
torch.save(model.state_dict(), 'model.pth')

# Evaluate the model
model.eval()
accuracy_metric = MulticlassJaccardIndex(num_classes=n_classes).to(device)

y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # Get the predicted class by finding the index with the maximum value in the class dimension
        preds = torch.argmax(outputs, dim=1)
        
        # Append predictions and true labels to lists
        y_pred.append(preds)
        y_true.append(torch.argmax(labels, dim=1))  # Get the true class from one-hot encoded labels

# Concatenate the lists into tensors
y_pred = torch.cat(y_pred, dim=0)
y_true = torch.cat(y_true, dim=0)

# Calculate IoU (Jaccard Index)
iou = accuracy_metric(y_pred, y_true)
print("Mean IoU =", iou.item())

test_img_number = 10
test_img = test_images[0]

test_img_cpy = test_img.clone().detach().cpu().numpy() # Has shape (1, 256, 256) but we need (256, 256, 1)
test_img_cpy = np.squeeze(test_img_cpy, axis=0)  # (256, 256)
test_img_cpy = np.expand_dims(test_img_cpy, axis=2)  # (256, 256, 1)

ground_truth = test_masks[0]
test_img_norm = test_img[:, :, 0][:, :, None]

# Ensure the image is on the correct device and shape
test_img_input = test_img.clone().detach().unsqueeze(0).to(device)

# Get prediction from the model
prediction = model(test_img_input)

# Since prediction is a tensor, use .cpu() to move it to CPU, and then use numpy for argmax
predicted_img = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

# Test image has to be copied to CPU and converted to numpy for plotting
test_img = test_img.cpu().numpy()

colormap = np.array([
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

predicted_img_color = colormap[predicted_img]
ground_truth_color = colormap[ground_truth]

# Plotting the results
plt.figure(figsize=(20, 10))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img_cpy[:, :, 0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth_color, cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img_color, cmap='jet')
plt.show()

# Save the predicted mask blended with the original image
alpha = 0.5
predicted_mask_image = Image.fromarray(predicted_img_color.astype('uint8'))
original_image = Image.fromarray(test_img_cpy[:, :, 0].astype('uint8')).convert('RGB')

predicted_mask_image = predicted_mask_image.resize(original_image.size)
blended_image = Image.blend(original_image, predicted_mask_image, alpha)
blended_image = blended_image.resize((512, 1024))
blended_image.save("predicted_mask_image.png")