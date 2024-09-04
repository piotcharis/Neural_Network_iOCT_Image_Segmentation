import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        
        mask = transforms.Resize((512, 256))(mask)
        mask = np.array(mask, dtype=np.int64)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
                
        return image, mask

transform = transforms.Compose([
    transforms.Resize((512, 256)),
    transforms.ToTensor()
])
