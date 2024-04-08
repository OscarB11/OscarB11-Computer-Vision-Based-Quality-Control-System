import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import textwrap
from IPython.display import display
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, AUC

import seaborn as sns



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

def preprocess_annotations(annotations, set_type):
    image_paths = []
    labels = []
    skipped_images = 0
    skipped_directories = {}
    
    for _, row in annotations.iterrows():
        image_path = os.path.join(dataset_dir, set_type, row['filename'])
        label = class_labels.get(row['class'], class_labels['no_defect'])
        
        if not os.path.exists(image_path):
            skipped_images += 1
            directory = os.path.dirname(image_path)
            skipped_directories[directory] = skipped_directories.get(directory, 0) + 1
            print(f"Skipped image; file not found: {image_path}")
        else:
            image_paths.append(image_path)
            labels.append(label)
    
    if skipped_images > 0:
        print(f"Skipped {skipped_images} images.")
        for directory, count in skipped_directories.items():
            print(f"Skipped directory: {directory}, Count: {count}")
    
    return image_paths, labels


# Load annotations from CSV files
train_annotations = pd.read_csv(os.path.join(dataset_dir, "train", "_annotations.csv"))
valid_annotations = pd.read_csv(os.path.join(dataset_dir, "valid", "_annotations.csv"))
test_annotations = pd.read_csv(os.path.join(dataset_dir, "test", "_annotations.csv"))

# Get image paths and labels for training, validation, and test sets
train_image_paths, train_labels = preprocess_annotations(train_annotations, "train")
valid_image_paths, valid_labels = preprocess_annotations(valid_annotations, "valid")
test_image_paths, test_labels = preprocess_annotations(test_annotations, "test")




# Define image dimensions
img_height = 300
img_width = 300
batch_size = 8
 
def load_and_preprocess_image(image_path, label):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        img = img / 255.0  # Normalize pixel values to [0, 1]
        
     
        return img, label
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None





# Creating TensorFlow datasets for the training, validation, and testing sets
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_labels))
valid_dataset = valid_dataset.map(load_and_preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Updated show_batch function to handle integer labels
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(len(image_batch)):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        # Get the corresponding class name for the label
        class_name = list(class_labels.keys())[list(class_labels.values()).index(label_batch[n])]
        plt.title(class_name)
        plt.axis('off')
    plt.show()

# Extract a batch and display it
image_batch, label_batch = next(iter(train_dataset))
show_batch(image_batch.numpy(), label_batch.numpy())


############################################################################################################
# Define the model architecture



model = tf.keras.Sequential([
    # First conv block
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Second conv block
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Third conv block
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Global Average Pooling
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Fully connected layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Output layer
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

# Compile the model
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




# Train the model
model.fit(
    train_dataset,
    epochs=2,
    validation_data=valid_dataset,
)



print("Testing accuracy:")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy:", test_accuracy, "\nTest Loss:", test_loss, "\n")



# Get predictions on the test dataset
test_predictions = model.predict(test_dataset)
test_predicted_labels = np.argmax(test_predictions, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.keys(), yticklabels=class_labels.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()







# Check unique labels in the training, validation, and test sets
unique_train_labels, train_label_counts = np.unique(train_labels, return_counts=True)
unique_valid_labels, valid_label_counts = np.unique(valid_labels, return_counts=True)
unique_test_labels, test_label_counts = np.unique(test_labels, return_counts=True)

print("Unique labels in training set:", unique_train_labels)
print("Label counts in training set:", train_label_counts)
print()
print("Unique labels in validation set:", unique_valid_labels)
print("Label counts in validation set:", valid_label_counts)
print()
print("Unique labels in test set:", unique_test_labels)
print("Label counts in test set:", test_label_counts)

class_names = list(class_labels.keys())


    
    
# Print out raw predictions for a few sample images
def debug_predictions(dataset, num_samples=5):
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(num_samples):
            print("Predictions for Sample", i+1)
            print("True Label:", class_names[labels[i]])
            print("Predicted Probabilities:", predictions[i])
            print("Predicted Class:", class_names[np.argmax(predictions[i])])
            print()

# Call the function for each dataset
debug_predictions(train_dataset)
print("\n\n")
debug_predictions(valid_dataset)
print("\n\n")

debug_predictions(test_dataset)
print("\n\n")

