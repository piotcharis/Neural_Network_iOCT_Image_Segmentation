# iOCT Image Segmentation
## General Description
This project was my Clinical Application Project as part of the Application Subject Medcine during my Computer Science Studies at the Technical University of Munich (TUM) and was done at the Chair for Computer Aided Medical Procedures & Augmented Reality in cooperation with ZEISS [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Zeiss_logo.svg/567px-Zeiss_logo.svg.png" alt="ZEISS" width="25" />](https://www.zeiss.com/meditec/de/home.html).

It focuses on segmenting retinal layers in intraoperative Optical Coherence Tomography (iOCT) volumes using a U-Net neural network model. It provides two main components:

1. Implementation: This section includes code for training the U-Net model using iOCT image data, allowing users to load, process, and train the model using a pre-structured dataset.

2. Testing App: A Flask-based web application for testing the trained model, making predictions on new images, and enabling users to upload a new model or revert to the default model.

The project is designed to facilitate iOCT image segmentation by classifying 13 retinal layers plus a background layer. The results are visualized as predicted masks overlaid on the original iOCT images.

## Implementation
### `main.py`
This file contains the data loading, processing and training functions for neural network model.

### `cap-oct-image-segmentation.ipynb`
This file is a Jupyter notebook that contains the same code as `main.py` in a more interactive format.

### `/data`
This directory contains the dataset used for training, validating and testing the neural network model. 

#### `/train` 
80% of the data in `/train` is used for training and 20% for validation

#### `/test`
These images are used for testing the model after training by visualizing the predicted masks vs. the manually segmented masks.

#### Data structure:
```
implementation
│
└───data
    │
    └───train
    │   │
    │   └───images
    │   │   │   image1.png
    │   │   │   ...
    │   │
    │   └───masks
    │       │   mask1.png
    │       │   ...
    │
    └───test
        │
        └───images
        │   │   image1.png
        │   │   ...
        │
        └───masks
            │   mask1.png
            │   ...
```

### Results
The final model was trained with 1300 iOCT slices for 250 Epochs. The result was:
```bash
Mean IoU = 0.6036953330039978
Accuracy = 0.7208034992218018
```
Accuracy is limited due to the data size, but the resulting masks are very reliable and in some cases more correct than the manually annotated masks.

## Testing App
### `app.py`
This file contains a simple Flask app that serves the web interface for testing the model, using the trained model from the `implementation` or any other uploaded model.
It includes the ability to make predictions, upload a new model, and revert to the default model.

#### Base URL
```arduino
http://localhost:<PORT>
```

The API is hosted locally on a specific port, which defaults to 5000 but can be configured using the PORT environment variable.

#### Endpoints:
1. Prediction Endpoint
   - Endpoint: `/predict`
   - Method: `POST`
   - Description: This endpoint accepts an image file, processes it, and returns a predicted segmentation mask overlaid on the input image.
   - Request:
        - Header: `Content-Type: multipart/form-data`
        - Body: `file`: Image file (Required). The image is expected to be in grayscale format.
   - Response:
        - Success:
            - Status Code: `200 OK`
            - Body: Returns a PNG image where the input image is overlaid with the predicted mask.
            - Content-Type: `image/png`
        - Failure:
            - Status Code: `400 Bad Request`
            - Body:
                ```json
                {
                    "error": "No file uploaded"
                } 
                ```
                
2. Upload New Model Endpoint
   - Endpoint: `/upload_model`
   - Method: `POST`
   - Description: Uploads a new U-Net model to replace the currently loaded model for predictions.
   - Request:
        - Header: `Content-Type: multipart/form-data`
        - Body: `model`: The new model file (Required). This file should be a .pth file containing the PyTorch model weights.
   - Response:
        - Success:
            - Status Code: `200 OK`
            - Body:
                ```json
                {
                    "message": "Model uploaded successfully"
                } 
                ```
        - Failure:
            - Status Code: `400 Bad Request`
            - Body:
                ```json
                {
                    "error": "No model file uploaded"
                } 
                ```
                
3. Revert to Default Model Endpoint
   - Endpoint: `/revert_model`
   - Method: `POST`
   - Description: Reverts the current model back to the default pre-loaded model. If a new model was uploaded, it is deleted from the system.
   - Request:
        - No body or parameters required.
   - Response:
        - Success:
            - Status Code: `200 OK`
            - Body:
                ```json
                {
                    "message": "Reverted to default model"
                } 
                ```
#### Additional Notes
1. Model Information:
    - The U-Net model used for predictions has 14 (13 retinal layers + 1 for the background) output classes and expects a grayscale input image with 1 channel.
    - The default model is loaded from model.pth at startup, and any uploaded model replaces this default until the API is reverted.

2. Image Preprocessing:
    - The image is resized to 256x512 before being passed to the model for prediction.
    - The model outputs a segmentation mask, which is colorized using a predefined color map.
      ```python
          np.array([
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
      ```

3. Device Handling:
    - The API automatically detects whether a CUDA-capable GPU is available and uses it if possible. Otherwise, it defaults to the CPU.

### `index.html`
This file contains the HTML code for the web interface of the testing app.

### `style.css`
This file contains the CSS code for styling the web interface.

## Usage
Make sure the data is structured as described above and train the model in one of two ways below. Both will save the trained model as a `.pth` file in both the `/implementation` as well as the `/testing_app` directories.

### 1. Way
Run the `cap-oct-image-segmentation.ipynb` file in a Jupyter notebook environment.

### 2. Way
Run the `main.py` file by running:
```bash
pip3 install -r requirements.txt
python main.py
```
### Test App
To use the trained model, you can run the Flask app by executing:
```bash
pip3 install -r requirements.txt
python app.py
```
Then, you can open the `index.html` file on a web browser to use the app.
