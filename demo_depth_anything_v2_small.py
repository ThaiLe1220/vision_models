from transformers import pipeline
from PIL import Image
import requests

# Create depth estimation pipeline
depth_estimator = pipeline(
    "depth-estimation", model="onnx-community/depth-anything-v2-small"
)

# Image URL
url = (
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"
)

# Load image
response = requests.get(url)
image = Image.open(requests.get(url, stream=True).raw)

# Predict depth of an image
depth = depth_estimator(image)

# Save the depth estimation output
depth_image = depth[0]["depth"]
depth_image.save("depth.png")
