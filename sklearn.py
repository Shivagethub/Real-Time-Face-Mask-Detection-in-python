import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model for mask detection
model = load_model('mask_detection.h5')

# Define a function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define a function to detect masks in the image
def detect_mask(img):
    img = preprocess_image(img)
    predictions = model.predict(img)
    return predictions[0][0] > 0.5
