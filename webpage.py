from flask import Flask, render_template, request, redirect, url_for
import os
import hashlib
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import VGG16
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('final.h5')

def calculate_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)


    img_final = preprocess_input(img_array)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Get features from an intermediate layer with a shape of (14, 14, 512)
    intermediate_layer_name = 'block4_pool'
    intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(intermediate_layer_name).output)
    features = intermediate_layer_model.predict(img_array)

    return features

def classify_image(image_path):
    # Preprocess the image using the same preprocessing steps as during training
    preprocessed_img = preprocess_image(image_path)

    # Predict the class probabilities
    predictions = model.predict(preprocessed_img)

    # Assuming it's a binary classification (Real or Fake)
    if predictions[0][0] > 0.7:
        return "Real"
    else:
        return "Fake"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Classify the image using your pre-trained model
        result = classify_image(file_path)

        if result == "Real":
            # Calculate the hash of the uploaded file only if it's authenticated as real
            file_hash = calculate_hash(file_path)
            return render_template('results.html', result=result, file_hash=file_hash, image_path=file_path)
        else:
            return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
