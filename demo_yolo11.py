"""Detection"""

# from ultralytics import YOLO
# from PIL import Image

# # Load a model
# model = YOLO("/home/ubuntu/Desktop/eugene/vision_models/models/yolo11n.pt")

# # Predict with the model
# results = model("images/image3.jpg")

# # Display or save the image with bounding boxes
# for result in results:
#     annotated_frame = result.plot()  # Annotated image (NumPy array)
#     image_with_boxes = Image.fromarray(annotated_frame)
#     image_with_boxes.show()  # Display the image


"""Segmentation"""

from ultralytics import YOLO
from PIL import Image, ImageDraw

# model = YOLO("/home/ubuntu/Desktop/eugene/vision_models/models/yolo11m-seg.pt")
# results = model("images/image9.jpg")  # Replace with your image path

tflite_model = YOLO(
    "/home/ubuntu/Desktop/eugene/vision_models/models/yolo11m-seg_saved_model/yolo11m-seg_float16.tflite"
)
results = tflite_model("images/image9.jpg")

# # Visualize the segmentation results
# for result in results:
#     # Get the annotated segmentation image as a NumPy array
#     annotated_frame = result.plot()  # Annotated image (NumPy array)

#     # Convert the annotated frame to an image
#     image_with_segmentation = Image.fromarray(annotated_frame)

#     # Show the image
#     image_with_segmentation.show()  # Open the image in the default image viewer


"""Classification"""
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch

# Paths
MODEL_PATH = "/home/ubuntu/Desktop/eugene/vision_models/models/yolo11m-cls.pt"
IMAGE_PATH = "images/money.jpg"
OUTPUT_PATH = "images/money_annotated.jpg"

# Load the classification model
model = YOLO(MODEL_PATH)

# Perform prediction
results = model(IMAGE_PATH)
