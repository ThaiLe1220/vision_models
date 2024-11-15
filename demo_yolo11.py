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

import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("vision_models/models/yolo11m-seg.pt")

# Directories
input_dir = "images"
output_dir = "images_segmentation"

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Iterate over all image files in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)

    # Check if the file is an image
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Perform segmentation
        results = model(input_path)

        # Print detailed information for each result
        for result in results:
            print(f"Processing file: {result.path}")
            print(f"Original image shape: {result.orig_shape}")
            print(f"Detection speed: {result.speed}")

            # Check if boxes exist
            if result.boxes is not None:
                print("Detected objects:")
                for box in result.boxes.data:
                    x1, y1, x2, y2, confidence, cls = box[:6]
                    label = result.names[int(cls)]
                    print(
                        f"  Label: {label}, Confidence: {confidence:.2f}, BBox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"
                    )

            # Check if masks exist
            if result.masks is not None:
                print("Segmentation masks available.")
                print(f"  Number of masks: {len(result.masks.data)}")

            # Additional attributes (keypoints, etc.)
            if result.keypoints is not None:
                print("Keypoints detected.")
            if result.obb is not None:
                print("Oriented bounding boxes detected.")

            print("\n")

        # Process and save the results for each image
        for result in results:
            # Filter objects with confidence > 0.5
            if result.boxes is not None:
                conf_mask = result.boxes.data[:, 4] > 0.5

                # Apply the confidence mask to boxes and masks
                result.boxes.data = result.boxes.data[conf_mask]
                if result.masks is not None:
                    result.masks.data = result.masks.data[conf_mask]

                if len(result.boxes.data) > 0:
                    # Get the annotated segmentation image as a NumPy array
                    annotated_frame = result.plot()

                    # Convert the annotated frame to an image
                    image_with_segmentation = Image.fromarray(annotated_frame)

                    # Save the image to the output directory
                    output_path = os.path.join(output_dir, f"segmented_{filename}")
                    image_with_segmentation.save(output_path)

                    # Optional: Show the image
                    image_with_segmentation.show()

print(f"Segmentation completed. Results saved in '{output_dir}'.")


"""Classification"""
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch

# # Paths
# MODEL_PATH = "models/yolo11m-cls.pt"
# IMAGE_PATH = "images/money.jpg"
# OUTPUT_PATH = "images/money_annotated.jpg"

# # Load the classification model
# model = YOLO(MODEL_PATH)

# # Perform prediction
# results = model(IMAGE_PATH)
