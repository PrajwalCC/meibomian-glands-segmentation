from flask import Flask, request, send_file
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load pre-trained model
model = load_model('models/gland-segmentation.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from POST request
    image_file = request.files['image']
    # print(request.files['hello'])

    # Read and preprocess image
    image = preprocess_image(image_file)

    # Make predictions
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # print(predictions.dtype)
    # print(predictions.shape)

    predictions_flat = predictions[0, :, :, 0]

# Convert to uint8 and scale to [0, 255]
    predictions_uint8 = (predictions_flat * 255).astype(np.uint8)

    # Create PIL image
    predicted_output_image = Image.fromarray(predictions_uint8)



    # Save the predicted output image to a BytesIO object
    img_byte_array = BytesIO()
    predicted_output_image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    # Return the image as response
    return send_file(img_byte_array, mimetype='image/png')

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
