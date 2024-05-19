from flask import Flask, request, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
CORS(app) 

# Load pre-trained models
glandModel = load_model('models/gland-segmentation.h5')
eyeLidModel = load_model('models/eyelid-segmentation.h5')

@app.route('/predict/gland', methods=['POST'])
def predict_gland():
    # Get image file from POST request
    image_file = request.files['image']

    # Read and preprocess image
    image = preprocess_image(image_file)

    # Make gland segmentation prediction
    gland_predictions = glandModel.predict(np.expand_dims(image, axis=0))
    gland_predictions_flat = gland_predictions[0, :, :, 0]

    # Convert to uint8 and scale to [0, 255]
    gland_predictions_uint8 = (gland_predictions_flat * 255).astype(np.uint8)

    # Create PIL image
    gland_output_image = Image.fromarray(gland_predictions_uint8)

    # Save the predicted output image to a BytesIO object
    gland_img_byte_array = BytesIO()
    gland_output_image.save(gland_img_byte_array, format='PNG')
    gland_img_byte_array.seek(0)

    # Return the gland image as response
    return send_file(gland_img_byte_array, mimetype='image/png')

@app.route('/predict/eyelid', methods=['POST'])
def predict_eyelid():
    # Get image file from POST request
    image_file = request.files['image']

    # Read and preprocess image
    image = preprocess_image(image_file)

    # Make eyelid segmentation prediction
    eyelid_predictions = eyeLidModel.predict(np.expand_dims(image, axis=0))
    eyelid_predictions_flat = eyelid_predictions[0, :, :, 0]

    # Convert to uint8 and scale to [0, 255]
    eyelid_predictions_uint8 = (eyelid_predictions_flat * 255).astype(np.uint8)

    # Create PIL image
    eyelid_output_image = Image.fromarray(eyelid_predictions_uint8)

    # Save the predicted output image to a BytesIO object
    eyelid_img_byte_array = BytesIO()
    eyelid_output_image.save(eyelid_img_byte_array, format='PNG')
    eyelid_img_byte_array.seek(0)

    # Return the eyelid image as response
    return send_file(eyelid_img_byte_array, mimetype='image/png')

def preprocess_image(image_file):
    # Read image file and convert to numpy array
    img = Image.open(image_file).convert('L')  # Convert to grayscale
    img = np.array(img)

    # Resize image to (256, 256)
    img = cv2.resize(img, (256, 256))

    # Normalize image
    img = img / 255.0

    return img

if __name__ == '__main__':
    app.run(debug=True)

# .\env\Scripts\activate     for activating env 