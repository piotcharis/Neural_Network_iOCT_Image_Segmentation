import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from PIL import Image

from data_loading import ImageSegmentationDataset, transform
from model import ImageSegmentationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = TensorBoardLogger('tb_logs', name='image_segmentation')

early_stop_callback = EarlyStopping(
    monitor='valid_loss',
    min_delta=0.0001,
    patience=10,
    verbose=True,
    mode='min'
)

def mask_to_color(mask):
    # Define the color map (you can customize this)
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

    # Ensure the mask values are within the range of the colormap
    mask = mask % len(colormap)
    color_mask = colormap[mask]

    return color_mask

def train() -> None:
    train_dataset = ImageSegmentationDataset('data/train', transform=transform)
    val_dataset = ImageSegmentationDataset('data/val', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    arch = 'unetplusplus'
    enc_name = 'efficientnet-b0'
    classes = 14

    model = smp.create_model(arch,
                            encoder_name = enc_name,
                            encoder_weights = "imagenet",
                            in_channels = 3,
                            classes = classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(device)
    cbs = L.callbacks.ModelCheckpoint(dirpath = f'./checkpoints_{arch}',
                                    filename = arch, 
                                    verbose = True, 
                                    monitor = 'valid_loss', 
                                    mode = 'min')
    
    pl_model = ImageSegmentationModel(model, optimizer, criterion)
    trainer = L.Trainer(accelerator='gpu', max_epochs=30, logger=logger, callbacks=[cbs, early_stop_callback], precision=16)
    trainer.fit(pl_model, train_loader, val_loader)

    torch.save(model.state_dict(), 'model.pth')
