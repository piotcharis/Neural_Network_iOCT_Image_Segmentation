import os
import sys
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, '../model')
from unet import UNet

# ---------------------------------------------------------------------------------------------- #

# Parameters
SIZE_X = 512    # Image height
SIZE_Y = 256    # Image width
n_classes = 14  # Number of classes for segmentation

# Paths
TRAIN_PATH_X = './data/train/images'
TRAIN_PATH_Y = './data/train/masks'
TEST_PATH_X = './data/test/images'
TEST_PATH_Y = './data/test/masks'

# Training parameters
num_epochs = 250
batch_size = 16 # Higher is better but requires more memory

# Set print options to display full arrays
np.set_printoptions(threshold=sys.maxsize)

# Define device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TensorBoard logger
logger = TensorBoardLogger('tb_logs', name='1')

# ---------------------------------------------------------------------------------------------- #

# Get image and mask file paths
train_ids_x = next(os.walk(TRAIN_PATH_X))[2]
train_ids_y = next(os.walk(TRAIN_PATH_Y))[2]
test_ids_x = next(os.walk(TEST_PATH_X))[2]
test_ids_y = next(os.walk(TEST_PATH_Y))[2]

#  Get and resize images
def get_resize_image(path):
    images = []
    for img_path in glob.glob(os.path.join(path, "*.png")): # Change the extension if needed
        img = cv2.imread(img_path, 0)                       # Read the image in grayscale mode    
        img = cv2.resize(img, (SIZE_Y, SIZE_X))             # Resize the image
        images.append(img)                                  # Append the image to the list
    return images                                # Convert the list to a NumPy array

# Get and resize masks
def get_resize_masks(path):
    masks = []
    for mask_path in glob.glob(os.path.join(path, "*.png")):
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)  # Prevent interpolation for the masks (nearest neighbor)
        masks.append(mask)
    return masks

train_images = np.array(get_resize_image(TRAIN_PATH_X)) # Convert the list to a NumPy array
train_masks = np.array(get_resize_masks(TRAIN_PATH_Y))

test_images = get_resize_image(TEST_PATH_X) # No need to convert to NumPy array, only for testing
test_masks = np.array(get_resize_masks(TEST_PATH_Y))

# Encode labels between 0 and n_classes, because PyTorch expects labels to be in this format for segmentation tasks 
def encode(labels):
    label_encoder = LabelEncoder()                                             # Initialize the label encoder
    n, h, w = labels.shape                                                     # Get the shape of the masks
    labels_reshaped = labels.reshape(-1, 1)                                    # Reshape the masks to 2 dimensions
    labels_reshaped_encoded = label_encoder.fit_transform(labels_reshaped)     # Fit and transform the labels to fit between 0 and n_classes
    labels_encoded_original_shape = labels_reshaped_encoded.reshape(n, h, w)   # Reshape the encoded labels back to the original shape
    return labels_encoded_original_shape

train_masks_encoded = encode(train_masks)
test_masks_encoded = encode(test_masks)

# Expand dimensions to add a channel dimension for images and masks and normalize the images between 0 and 1
train_images = np.expand_dims(train_images, axis=3)
train_images = train_images / 255.0  # Normalization

test_images = np.expand_dims(test_images, axis=3)
test_images = test_images / 255.0  # Normalization

train_masks_input = np.expand_dims(train_masks_encoded, axis=3)
test_masks_input = np.expand_dims(test_masks_encoded, axis=3)

# Split the data into training and testing sets (90% training, 10% testing)
x_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)

# Convert labels to one-hot encoded format to match the output of the model (n_classes)
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)].reshape((*y.shape[:-1], num_classes)) # eye creates an identity matrix of shape (num_classes, num_classes) and then reshapes it to the shape 
                                                                                    # of y with num_classes columns to get one-hot encoded labels
                                                                                    # *y.shape[:-1] is (256, 256) and y.reshape(-1) reshapes it to a 1D array
                                                                                    
train_masks_one_hot = one_hot_encode(y_train, num_classes=n_classes)
y_train_one_hot = train_masks_one_hot.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes)) # Reshape the one-hot encoded labels to the original shape

test_masks_one_hot = one_hot_encode(y_test, num_classes=n_classes)
y_test_one_hot = test_masks_one_hot.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes)) # Reshape the one-hot encoded labels to the original shape

# Convert the data to PyTorch tensors, permute the dimensions to match the format (N, C, H, W), and move them to the device
x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2).to(device) 
y_train_one_hot = torch.tensor(y_train_one_hot, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
y_test_one_hot = torch.tensor(y_test_one_hot, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

# ---------------------------------------------------------------------------------------------- #

# UNet model

# ---------------------------------------------------------------------------------------------- #

# Initialize the model and move it to the device
model = UNet(n_classes=n_classes, img_channels=1).to(device) 

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss for multi-class segmentation

train_loader = DataLoader(list(zip(x_train, y_train_one_hot)), batch_size=batch_size, shuffle=True) # DataLoader for training, shuffle the data for better training and generalization
test_loader = DataLoader(list(zip(X_test, y_test_one_hot)), batch_size=batch_size, shuffle=False)   # DataLoader for testing

train_losses, val_losses = [], [] # Lists to store training and validation losses

trainer = pl.Trainer(max_epochs=num_epochs, logger=logger) # PyTorch Lightning Trainer

trainer.fit(model, train_loader, test_loader) # Fit the model
    
# Save the model
torch.save(model.state_dict(), 'model.pth')

# ---------------------------------------------------------------------------------------------- #

# Put the model in evaluation mode and calculate the mean IoU (Jaccard Index) and Accuracy on the test set
model.eval().to(device)
jaccard_idx = MulticlassJaccardIndex(num_classes=n_classes).to(device)
accuracy = MulticlassAccuracy(num_classes=n_classes).to(device)

y_pred = []
y_true = []

# Get predictions and true labels for the test set and calculate the mean IoU
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)                     # Get the predictions
        preds = torch.argmax(outputs, dim=1)        # Get the predicted class by finding the index with the maximum value in the class dimension
        y_pred.append(preds)                        # Append the predictions to the list
        y_true.append(torch.argmax(labels, dim=1))  # Get the true class from the one-hot encoded labels

# Concatenate the lists into tensors
y_pred = torch.cat(y_pred, dim=0)
y_true = torch.cat(y_true, dim=0)

# Calculate and print the IoU (Jaccard Index) & Accuracy
iou = jaccard_idx(y_pred, y_true)
acc = accuracy(y_pred, y_true)
print("Mean IoU =", iou.item())
print("Accuracy =", acc.item())

# ---------------------------------------------------------------------------------------------- #

# Get the first test image
# TODO: Add a loop to get multiple test images
test_img = test_images[0]

# Save a copy of the test image for plotting and convert it to a NumPy array with the correct shape
test_img_cpy = test_img.clone().detach().cpu().numpy()
test_img_cpy = np.squeeze(test_img_cpy, axis=0)         # Remove the batch dimension
test_img_cpy = np.expand_dims(test_img_cpy, axis=2)     # Add the channel dimension

# Get the first test mask (ground truth)
ground_truth = test_masks[0]

# Normalize the test image to be between 0 and 1 and add a channel dimension
test_img_norm = test_img[:, :, 0][:, :, None] 

# Ensure the image is on the correct device and shape
test_img_input = test_img.clone().detach().unsqueeze(0).to(device)

# Get prediction from the model
prediction = model(test_img_input)

# Get the predicted mask by finding the index with the maximum value in the class dimension
# and convert it to a NumPy array
predicted_img = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

# Test image has to be moved to the CPU and converted to numpy for plotting
test_img = test_img.cpu().numpy()

# Color map for the masks with the RGB values used in DeepVision
color_map = np.array([
        [0, 0, 0],       # Background
        [229, 4, 2],     # ILM
        [49, 141, 171],  # RNFL
        [138, 61, 199],  # GCL
        [154, 195, 239], # IPL
        [245, 160, 56],  # INL
        [232, 146, 141], # OPL
        [245, 237, 105], # ONL
        [232, 206, 208], # ELM
        [128, 161, 54],  # PR
        [32, 207, 255],  # RPE
        [232, 71, 72],   # BM
        [212, 182, 222], # CC
        [196, 45, 4],    # CS
])

# Apply the color map to the predicted and ground truth masks
predicted_img_color = color_map[predicted_img]
ground_truth_color = color_map[ground_truth]

# Plot the predicted blended image
plt.figure(figsize=(20, 10))
alpha = 0.5
plt.title('Blended Prediction')
plt.imshow(test_img_cpy[:, :, 0], cmap='gray')
plt.imshow(predicted_img_color, cmap='jet', alpha=alpha)

# Save the blended plot to a file
plt.savefig('prediction.png')

plt.show()

# Plot the three seperate images
plt.figure(figsize=(20, 10))

# Test image
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img_cpy[:, :, 0], cmap='gray')

# Ground truth mask
plt.subplot(232)
plt.title('Ground Truth')
plt.imshow(ground_truth_color, cmap='jet')

# Predicted mask
plt.subplot(233)
plt.title('Model Prediction')
plt.imshow(predicted_img_color, cmap='jet')
plt.show()
