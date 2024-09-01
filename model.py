import torch
from torchvision import transforms
from PIL import Image
import os

# Assuming the LitSegmentation and CustomSegmentationDataset classes are already defined

def load_model(model_path):
    trained_model = LitSegmentation()
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()  # Set the model to evaluation mode
    return trained_model

def predict(trained_model, image_path):
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    # Run the model
    with torch.no_grad():  # Disable gradient calculation
        output = trained_model(image)['out']

    # Get the predicted mask (assuming you want the class with the highest score)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0)

    return predicted_mask

if __name__ == "__main__":
    # Load the trained model
    model = load_model("model.pth")

    # Specify the path to a new image
    image_path = "./data/test/images/000003_d19dd6c3-895f-4f90-bdc6-0006b85b0c7c.png"

    # Make a prediction
    predicted_mask = predict(model, image_path)

    # Convert the predicted mask to a PIL image (optional)
    predicted_mask_pil = transforms.ToPILImage()(predicted_mask.byte())

    # Save or display the predicted mask
    predicted_mask_pil.save("predicted_mask.png")
    predicted_mask_pil.show()
