from ultralytics import YOLO
from PIL import Image

# Load the YOLO11 model
model = YOLO("/home/ubuntu/Desktop/eugene/vision_models/models/yolo11m.pt")

# # Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'
