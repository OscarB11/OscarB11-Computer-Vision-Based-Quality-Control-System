import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

# Load the InceptionV3 model pre-trained on ImageNet data
print("Loading InceptionV3 model...")
model = InceptionV3(weights='imagenet')
print("InceptionV3 model loaded successfully.")

# Load and preprocess an image
img_path = "boo12/pcb-defect-dataset/test/images/l_light_01_missing_hole_04_2_600.jpg"
print("Loading and preprocessing the image...")
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print("Image loaded and preprocessed successfully.")

# Make predictions with InceptionV3
print("Making predictions with InceptionV3 model...")
preds = model.predict(x)
# Decode the predictions
decoded_preds = decode_predictions(preds, top=3)[0]
print('InceptionV3 Predictions:', decoded_preds)

# Load the ResNet50 model pre-trained on ImageNet data
print("\nLoading ResNet50 model...")
model = ResNet50(weights='imagenet')
print("ResNet50 model loaded successfully.")

# Load and preprocess an image for ResNet50
print("Loading and preprocessing the image for ResNet50...")
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print("Image loaded and preprocessed successfully.")

# Make predictions with ResNet50
print("Making predictions with ResNet50 model...")
preds = model.predict(x)
# Decode the predictions
decoded_preds = decode_predictions(preds, top=3)[0]
print('ResNet50 Predictions:', decoded_preds)

