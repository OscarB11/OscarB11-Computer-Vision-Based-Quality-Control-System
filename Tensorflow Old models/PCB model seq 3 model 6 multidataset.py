import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print("\n test \n ",e)


# Correctly define your dataset directory (use a raw string for Windows paths) 
dataset_dir = r"boo12\PCB Defects.v1i.tensorflow"



# Map your class labels
class_labels = {
    "open_circuit": 0,
    "missing_hole": 1,
    "spurious_copper": 2,
    "short": 3,
    "mouse_bite": 4,
    "spur": 5,
    "no_defect": 6  # Adding a class for no defect
}


# Path to dataset
dataset_dir = "boo12\PCB Defects.v1i.tensorflow"

# Image dimensions and batch size
IMG_SIZE = (224, 224)  # Image size that MobileNetV2 expects
BATCH_SIZE = 8

# Data augmentation and preprocessing for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Just rescaling for validation and test sets
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Data loaders
train_generator = train_datagen.flow_from_directory(
    f'{dataset_dir}/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = test_val_datagen.flow_from_directory(
    f'{dataset_dir}/valid',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_val_datagen.flow_from_directory(
    f'{dataset_dir}/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)  # No need to shuffle the test set

# Model architecture using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # You can adjust the number of units
predictions = Dense(len(class_labels), activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=1,  # Adjust as per requirement
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE)

filenames = [
    r"boo12\PCB Defects.v1i.tensorflow\test\missing_hole\l_light_06_missing_hole_01_1_600_jpg.rf.6b0757887b56e9a02664eb7e19d46d78.jpg",
]

# Function to preprocess and predict the class of an image
def predict_image_class(model, img_path):
    # Load the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model input
    img_array_expanded = np.expand_dims(img_array, axis=0)
    # Preprocess the image
    img_preprocessed = img_array_expanded / 255.
    # Predict the class
    prediction = model.predict(img_preprocessed)
    # Return the index of the class with the highest probability
    return np.argmax(prediction), np.max(prediction)

# Iterate over the filenames, predict the class, and print the results
# Initialize counters
total_images = len(filenames)
passed_images = 0

# Iterate over the filenames, predict the class, and print the results
for filename in filenames:
    predicted_class_index, confidence = predict_image_class(model, filename)
    # Assuming class_labels is a dictionary mapping class indices to class names
    predicted_class_name = [key for key, value in class_labels.items() if value == predicted_class_index][0]
    print(f"Image: {filename}, Predicted Class: {predicted_class_name}, Confidence: {confidence:.2f}")
    
    # Check if the predicted class is "no_defect"
    if predicted_class_name == "no_defect":
        passed_images += 1

# Calculate the percentage of passed images
pass_percentage = (passed_images / total_images) * 100

# Print the number of passed images, total images, and pass percentage
print(f"Number of passed images: {passed_images}")
print(f"Total number of images: {total_images}")
print(f"Pass percentage: {pass_percentage:.2f}%")



# Assuming test_generator is defined somewhere in your code
print("Generating predictions for the test set...")
test_generator.reset()  # Ensuring the generator is starting from the beginning
predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)

# Get the filenames from the test generator
filenames = test_generator.filenames

# Assuming class_labels is defined somewhere in your code
# Assuming predictions are probabilities (one-hot encoded)
predicted_class_indices = np.argmax(predictions, axis=1)
predicted_class_names = [list(class_labels.keys())[i] for i in predicted_class_indices]

# Print filenames alongside predicted class names
for filename, predicted_class_name in zip(filenames, predicted_class_names):
    print(f"Filename: {filename}, Predicted Class: {predicted_class_name}")

# Assuming true_labels is defined somewhere in your code
true_labels = test_generator.classes

# Confusion matrix
conf_mat = confusion_matrix(true_labels, np.argmax(predictions, axis=1), normalize='true')  # Normalized

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt=".2f", xticklabels=list(class_labels.keys()), yticklabels=list(class_labels.keys()))
plt.title('Confusion Matrix (Normalized)')
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Calculate the number of correctly predicted samples
correct_predictions = np.sum(predicted_class_indices == true_labels)

# Calculate the total number of samples
total_samples = len(true_labels)

# Calculate the accuracy
accuracy = correct_predictions / total_samples

# Print the number of correctly predicted samples, total samples, and accuracy
print(f"Number of correctly predicted samples: {correct_predictions}")
print(f"Total number of samples: {total_samples}")
print(f"Accuracy: {accuracy * 100:.2f}%")