import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(0)

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

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Draw rectangles around the faces and detect masks
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        has_mask = detect_mask(roi)
        color = (0, 255, 0) if has_mask else (0, 0, 255)
        label = 'Mask' if has_mask else 'No Mask'
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the frame with face detection and mask detection results
    cv2.imshow('Face Mask Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

