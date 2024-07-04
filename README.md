# CIFAR-10 Image Classification Web App

This web application allows users to upload images and classify them using a CIFAR-10 trained model.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the source code.

2. Navigate to the project directory:

3. Install the required packages: pip install -r requirements.txt

## Running the Application

1. Start the Flask server: python app.py

2. Go to the generated URL.

You should now see the CIFAR-10 Image Classification web interface.

## Usage

1. Click the "Select Image" button to choose an image file.
2. Once an image is selected, click "Classify Image".
3. The application will process the image and display the classification result.

## Troubleshooting

- If you encounter any issues with dependencies, make sure you've installed all required packages from the `requirements.txt` file.
- Ensure that the `cifar_net.pth` model file is in the same directory as `app.py`.
- If you're having trouble with the port number, you can modify it in the `app.py` file.

## License

[MIT License](LICENSE)

## Acknowledgments

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch: https://pytorch.org/
