import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(0)
