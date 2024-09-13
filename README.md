# OCT Image Segmentation
## Implementation
### `main.py`
This file contains the data loading, processing and training functions for neural network model.

### `cap-oct-image-segmentation.ipynb`
This file is a Jupyter notebook that contains the same code as `main.py` in a more interactive format.

### `data`
This directory contains the dataset used for training, validating and testing the neural network model. 
The dataset consists of OCT images provided by Zeiss and their corresponding masks created on the Deepvision platform.

#### Data structure:
```
data
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


## Testing App
### `app.py`
This file contains a simple Flask app that serves the simple web interface for testing the model, using the trained model from the `implementation` or an uploaded model.

### `index.html`
This file contains the HTML code for the web interface of the testing app.

### `style.css`
This file contains the CSS code for styling the web interface.

## Usage
To train the model, you can run the `cap-oct-image-segmentation.ipynb` file in a Jupyter notebook environment. The notebook contains the code for loading the dataset, preprocessing the data, training the model and evaluating the model and will save the trained model to a file.

To use the model, you can run the Flask app by running:
```bash
pip3 install -r requirements.txt
python app.py
```
Then, you can open a web browser and open the `index.html` file to use the testing app.
