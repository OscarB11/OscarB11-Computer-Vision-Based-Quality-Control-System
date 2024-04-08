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



"""



def get_random_images(dataset, num_images=10):
    random_images = []
    seen_images = set()
    for _, batch in enumerate(dataset):
        images, labels, filenames = batch
        for i in range(len(images)):
            image_tensor = images[i].numpy()
            label = labels[i].numpy()
            filename = filenames[i].numpy().decode('utf-8')
            image_hash = hash(image_tensor.tobytes())
            if image_hash not in seen_images:
                seen_images.add(image_hash)
                random_image = (image_tensor, label, filename)
                random_images.append(random_image)
                if len(random_images) == num_images:
                    return random_images

# Get random images from the training set
random_train_images = get_random_images(train_dataset)

# Get random images from the validation set
random_valid_images = get_random_images(valid_dataset)

# Get random images from the test set
random_test_images = get_random_images(test_dataset)





# Display random images from the training set
fig, axs = plt.subplots(2, 5, figsize=(20, 10), dpi=100)  # Increase figure size for more space
axs = axs.ravel()
for i, (image, label, filename) in enumerate(random_train_images):
    axs[i].imshow(image)
    # Wrap titles more aggressively to ensure they fit, and reduce font size if necessary
    axs[i].set_title('\n'.join(textwrap.wrap(f"Label: {list(class_labels.keys())[list(class_labels.values()).index(label)]} (File: {filename})", 30)), fontsize=7)
    axs[i].axis('off')
# Increase hspace for more vertical spacing between rows
plt.subplots_adjust(wspace=0.5, hspace=1, left=0.05, right=0.95, bottom=0.05, top=0.95)
plt.show()



# Display random images from the validation set
fig, axs = plt.subplots(2, 5, figsize=(15, 8), dpi=100)
axs = axs.ravel()
for i, (image, label, filename) in enumerate(random_valid_images):
    axs[i].imshow(image)
    axs[i].set_title('\n'.join(textwrap.wrap(f"Label: {list(class_labels.keys())[list(class_labels.values()).index(label)]} (File: {filename})", 50)), fontsize=8)
    axs[i].axis('off')
    plt.subplots_adjust(wspace=0.5, hspace=0.7, left=0.1, right=0.9, bottom=0.1, top=0.9)
plt.show()

# Display random images from the test set
fig, axs = plt.subplots(2, 5, figsize=(15, 8), dpi=100)
axs = axs.ravel()
for i, (image, label, filename) in enumerate(random_test_images):
    axs[i].imshow(image)
    axs[i].set_title('\n'.join(textwrap.wrap(f"Label: {list(class_labels.keys())[list(class_labels.values()).index(label)]} (File: {filename})", 50)), fontsize=8)
    axs[i].axis('off')
plt.subplots_adjust(wspace=0.4, hspace=0.6, left=0.1, right=0.9, bottom=0.1, top=0.9)
plt.show()

"""
############################################################################################################
# Define the model architecture

num_classes = len(class_labels)  # This will set num_classes to 7 based on your class_labels dictionary

model = tf.keras.Sequential([
    # Initial Convolution and pooling to reduce spatial dimensions
    tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
    
    # Block1 - repeated Conv operations followed by concatenation
    *[tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
    ]) for _ in range(6)],
    
    # Transition Layer
    tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
    ]),
    
    # Block2
    *[tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
    ]) for _ in range(6)],
    
    # Classification Layer
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Train the model with callbacks
history = model.fit(
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

