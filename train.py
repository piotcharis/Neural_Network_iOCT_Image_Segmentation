import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp

from data_loading import ImageSegmentationDataset, transform
from model import ImageSegmentationModel

CLASSES = ['Background', 'ILM', 'RNFL', 'GCL', 'IPL', 'INL', 'OPL', 'ONL', 'ELM', 'PR', 'RPE', 'BM', 'CC', 'CS']
ENCODER_WEIGHTS = 'imagenet'
ENCODER = 'resnet50'
ACTIVATION = None
MULTICLASS_MODE : str = "multiclass"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = TensorBoardLogger('tb_logs', name='image_segmentation')

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=True,
    mode='min'
)

def train() -> None:
    dataset = ImageSegmentationDataset('data', transform=transform) # Create dataset
    train_size = int(0.8 * len(dataset)) # Split dataset into train and validation sets (80% train, 20% validation)
    val_size = len(dataset) - train_size # Calculate validation set size
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) # Split dataset into train and validation sets using random_split
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    arch = 'unet'
    enc_name = 'resnet50'
    classes = 14

    model = smp.create_model(arch,
                            encoder_name = enc_name,
                            encoder_weights = "imagenet",
                            in_channels = 3,
                            classes = classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(device)
    cbs = L.callbacks.ModelCheckpoint(dirpath = f'./checkpoints_{arch}',
                                    filename = arch, 
                                    verbose = True, 
                                    monitor = 'valid_loss', 
                                    mode = 'min')
    
    pl_model = ImageSegmentationModel(model, optimizer, criterion)
    trainer = L.Trainer(accelerator='gpu', max_epochs=100, logger=logger, callbacks=cbs)
    trainer.fit(pl_model, train_loader, val_loader)

   
    
    # trainer = L.Trainer(max_epochs=50, logger=logger, callbacks=[early_stop_callback])
    # trainer.fit(model, train_loader, val_loader)
    
    torch.save(model.state_dict(), 'model.pth')
