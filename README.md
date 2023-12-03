# Blockchain-based-antique-verification-system

# https://youtu.be/IRB60VkzdFQ

## Overview

This project implements an antique authenticity detection system,specifically images using a combination of a pre-trained convolutional neural network (VGG16) for feature extraction and a custom neural network for classification. The system can distinguish between real and AI-generated images.

## Features

- **Image Upload:** Users can upload images through a web interface.
- **Classification:** The system classifies uploaded images as "Real" or "Fake" using a pre-trained model.
- **Hashing:** For authenticated real images, the system calculates and displays the SHA-256 hash.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- PIL (Pillow)
- Numpy

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/blockchain-based-antique-verification.git
   ```

2. Change into the project directory:

   ```bash
   cd blockchain-based-antique-verification
   ```

3. Run the Flask application:

   ```bash
   python webpage.py
   ```

   The application will be accessible at `http://localhost:5001`.

## Usage

1. Open the application in a web browser.

2. Upload an image using the provided form.

3. The system will classify the image as "Real" or "Fake."

4. For authenticated real images, the SHA-256 hash is displayed.





## Acknowledgments

- The pre-trained VGG16 model is part of the TensorFlow/Keras library.
- Special thanks to contributors and open-source community.

