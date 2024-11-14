# import onnxruntime as ort
# import numpy as np
# from PIL import Image

# # Path to your ONNX model and test image
# MODEL_PATH = "MobileNet-v3-Large-Quantized.onnx"
# IMAGE_PATH = "test_image.jpg"


# # Load and preprocess the image
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("RGB")
#     img = img.resize((224, 224))  # MobileNet typically uses 224x224
#     img_data = np.array(img).astype("float32") / 255.0  # Normalize to [0,1]
#     img_data = (img_data - 0.5) / 0.5  # Normalize to [-1,1] if required
#     img_data = np.transpose(img_data, (2, 0, 1))  # Change to CxHxW
#     img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
#     return img_data


# # Load the model
# def load_model(model_path):
#     session = ort.InferenceSession(model_path)
#     return session


# # Run inference
# def run_inference(session, input_data):
#     input_name = session.get_inputs()[0].name
#     outputs = session.run(None, {input_name: input_data})
#     return outputs


# # Main function
# def main():
#     input_data = preprocess_image(IMAGE_PATH)
#     session = load_model(MODEL_PATH)
#     outputs = run_inference(session, input_data)
#     print("Model output:", outputs)


# if __name__ == "__main__":
#     main()


print("Hello World")
