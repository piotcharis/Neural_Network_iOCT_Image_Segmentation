import torch
from PIL import Image
from torchvision import transforms
from data_loading import ImageSegmentationDataset
from model import ImageSegmentationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    image.save('sample_image.png')
    mask.save('predicted_mask.png')
