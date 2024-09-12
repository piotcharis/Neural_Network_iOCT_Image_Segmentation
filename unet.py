import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

class UNet(pl.LightningModule):
    def __init__(self, n_classes, img_channels):
        super(UNet, self).__init__()

        # Contraction path (encoder)
        # It takes the input image and reduces its spatial dimensions while increasing the number of channels.
        # This is done by applying a series of convolutions and pooling layers.
        # The output of each convolutional layer is passed through a ReLU activation function, which introduces non-linearity.
        # The output of each pooling layer is passed to the next layer and the output of the last layer is the input to the expansive path.
        
        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, padding=1) # The first convolutional layer takes the input image and applies 16 filters of size 3x3.
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)         # The second convolutional layer takes the output of the first layer and applies 16 filters of size 3x3.
        self.dropout1 = nn.Dropout(0.1)                                    # The dropout layer is used to prevent overfitting by randomly setting a fraction of the input units to zero.
        self.pool1 = nn.MaxPool2d(2, 2)                                    # The pooling layer reduces the spatial dimensions of the input by taking the maximum value in each 2x2 region.

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)          # The third convolutional layer takes the output of the pooling layer and applies 32 filters of size 3x3 and so on.
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

        # Expansive path (decoder)
        # It takes the output of the last convolutional layer in the contraction path and increases the spatial dimensions while reducing the number of channels.
        # This is done by applying a series of transposed convolutions and concatenating the output of each transposed convolution with the output of the corresponding 
        # convolution in the contraction path.
        # The output of each transposed convolution is passed through a ReLU activation function, which introduces non-linearity.
        # The output of the last transposed convolution is the final output of the network.

        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # The first transposed convolutional layer takes the output of the last convolutional layer in the contraction path and applies 128 filters of size 2x2.
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)       # The second convolutional layer takes the output of the transposed convolutional layer and applies 128 filters of size 3x3.
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)     # The third convolutional layer takes the output of the second convolutional layer and applies 128 filters of size 3x3.
        self.dropout6 = nn.Dropout(0.2)                                  # The dropout layer is used to prevent overfitting by randomly setting a fraction of the input units to zero.

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

    # Computations performed at each forward pass of the network.
    def forward(self, x):
        # Contraction path
        c1 = F.relu(self.conv1(x))      # The input image is passed through the first convolutional layer and the output is passed through a ReLU activation function.
        c1 = F.relu(self.conv1_2(c1))   # The output of the first convolutional layer is passed through the second convolutional layer and the output is passed through a ReLU activation function.
        c1 = self.dropout1(c1)          # The output of the second convolutional layer is passed through a dropout layer.
        p1 = self.pool1(c1)             # The output of the dropout layer is passed through a pooling layer.

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
        u6 = self.up6(c5)               # The output of the last convolutional layer in the contraction path is passed through the first transposed convolutional layer.
        u6 = torch.cat([u6, c4], dim=1) # The output of the transposed convolutional layer is concatenated with the output of the corresponding convolution in the contraction path.
        c6 = F.relu(self.conv6(u6))     # The concatenated output is passed through the second convolutional layer and the output is passed through a ReLU activation function.
        c6 = F.relu(self.conv6_2(c6))   # The output of the second convolutional layer is passed through the third convolutional layer and the output is passed through a ReLU activation function.
        c6 = self.dropout6(c6)          # The output of the third convolutional layer is passed through a dropout layer.

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

        outputs = self.conv10(c9)      # The output of the last convolutional layer is passed through a convolutional layer with a kernel size of 1x1 to produce the final output of the network.

        return outputs

    # The training step is used to compute the training loss and update the weights of the network.
    def training_step(self, batch, batch_idx):
        x, y = batch                     # The input image and the target mask are unpacked from the batch.
        y_hat = self(x)                  # The input image is passed through the network to produce the output mask.
        y = y.squeeze(-1)                # The target mask is reshaped to match the output mask.
        loss = F.cross_entropy(y_hat, y) # The cross-entropy loss is computed between the output mask and the target mask.
        self.log("train_loss", loss)     # The loss is logged for visualization.
        return loss
    
    # The validation step is used to compute the validation loss.
    def validation_step(self, batch, batch_idx):
        x,y = batch                           # The input image and the target mask are unpacked from the batch.
        y_hat = self(x)                       # The input image is passed through the network to produce the output mask.
        y = y.squeeze(-1)                     # The target mask is reshaped to match the output mask.
        val_loss = F.cross_entropy(y_hat, y)  # The cross-entropy loss is computed between the output mask and the target mask.
        self.log("val_loss", val_loss)        # The loss is logged for visualization.
        return val_loss                         
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)