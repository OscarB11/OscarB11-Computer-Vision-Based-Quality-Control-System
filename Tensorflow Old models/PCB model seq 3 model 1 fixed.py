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


# Function to count the number of images in each directory
def count_images(directory):
    image_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_count += 1
    return image_count

# Display the number of images in each directory
train_dir = os.path.join(dataset_dir, "train")
valid_dir = os.path.join(dataset_dir, "valid")
test_dir = os.path.join(dataset_dir, "test")

print(f"Number of images in training directory: {count_images(train_dir)}")
print(f"Number of images in validation directory: {count_images(valid_dir)}")
print(f"Number of images in test directory: {count_images(test_dir)}")


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


# Function to randomly sample and display image paths and labels
def display_random_samples(image_paths, labels, num_samples=5):
    # Set pandas to display the full path without truncation
    pd.set_option('display.max_colwidth', None)

    sample_indices = np.random.choice(len(image_paths), num_samples, replace=False)
    sample_paths = np.array(image_paths)[sample_indices]
    sample_labels = np.array(labels)[sample_indices]
    sample_data = pd.DataFrame({
        'Image Path': sample_paths,
        'Label': sample_labels
    })
    display(sample_data)

print("\nTraining Dataset Samples:\n")
display_random_samples(train_image_paths, train_labels)

print("\nValidation Dataset Samples:\n")
display_random_samples(valid_image_paths, valid_labels)

print("\nTest Dataset Samples:\n")
display_random_samples(test_image_paths, test_labels)


# Define image dimensions
img_height = 200
img_width = 200
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


def display_batch(dataset, filenames, num_images=8, dataset_name=''):
    # Get a batch of images and labels
    image_batch, label_batch = next(iter(dataset))
    
    # Calculate number of rows and columns for the subplots based on the num_images
    cols = int(np.sqrt(num_images))
    rows = num_images // cols + int(num_images % cols > 0)

    # Display the images and labels
    plt.figure(figsize=(cols * 5, rows * 5))  # Increase the size for clarity
    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)
        img = image_batch[i].numpy()  # Convert the tensor to a numpy array
        plt.imshow(img)
        
        # Get image dimensions
        img_height, img_width, _ = img.shape
        
        filename = filenames[i].numpy().decode('utf-8').split('/')[-1]
        # Wrap the filename to the next line if it's too long
        wrapped_filename = '\n'.join(textwrap.wrap(filename, width=25))
        
        # Append the image size to the title
        title = f"{dataset_name}\nLabel: {label_batch[i]}\nSize: {img_width}x{img_height}\nFilename: {wrapped_filename}"
        plt.title(title, fontsize=8)
        plt.axis("off")
    plt.tight_layout()  # Adjust the layout for better visibility
    plt.show()
    #print(image_batch[0].numpy())


train_filenames = tf.convert_to_tensor(train_image_paths)

# Use the improved function to display the images from the training dataset
print("\nDisplaying a batch of images from the training dataset:")

#display_batch(train_dataset, train_filenames, dataset_name='Training Set')

print("\nDisplaying a batch of images from the valid dataset:")

#display_batch(valid_dataset, tf.convert_to_tensor(valid_image_paths), dataset_name='Validation Set')

print("\nDisplaying a batch of images from the test dataset:")

#display_batch(test_dataset, tf.convert_to_tensor(test_image_paths), dataset_name='Test Set')

















############################################################################################################
# Define the model architecture






model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),  # Dropout layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Regularization
    tf.keras.layers.Dropout(0.5),  # Dropout layer
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])



# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Train the model

model.fit(
    train_dataset,
    epochs=1,
    validation_data=valid_dataset,
)






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














def display_batch_with_predictions(dataset, filenames, model, class_names, num_images=8, dataset_name=''):
    # Get a batch of images and labels
    image_batch, label_batch = next(iter(dataset))
    
    # Make predictions on the image batch
    predictions = model.predict(image_batch)
    
    # Calculate number of rows and columns for the subplots based on the num_images
    cols = int(np.sqrt(num_images))
    rows = num_images // cols + int(num_images % cols > 0)

    # Display the images, predictions, and labels
    plt.figure(figsize=(cols * 5, rows * 5))  # Increase the size for clarity
    for i in range(min(num_images, len(image_batch))):
        ax = plt.subplot(rows, cols, i + 1)
        img = image_batch[i].numpy()  # Convert the tensor to a numpy array
        plt.imshow(img)
        
        # Get the predicted class with highest probability
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[label_batch[i].numpy()]
        
        filename = filenames[i].numpy().decode('utf-8').split('/')[-1]
        # Wrap the filename to the next line if it's too long
        wrapped_filename = '\n'.join(textwrap.wrap(filename, width=25))
        
        # Append the image size to the title
        title = f"{dataset_name}\nTrue: {true_class}\nPred: {predicted_class}\nFilename: {wrapped_filename}"
        plt.title(title, fontsize=8)
        plt.axis("off")
    plt.tight_layout()  # Adjust the layout for better visibility
    plt.show()

# Example usage:
# Assuming train_dataset, valid_dataset, and test_dataset are your datasets
# and train_image_paths, valid_image_paths, test_image_paths are the paths to the images in these datasets

# You need to adjust `class_labels` to your specific class names
class_names = list(class_labels.keys())

# Convert your file paths to tensors
train_filenames = tf.convert_to_tensor(train_image_paths)
valid_filenames = tf.convert_to_tensor(valid_image_paths)
test_filenames = tf.convert_to_tensor(test_image_paths)

#print("\nDisplaying a batch of images with predictions from the training dataset:")
#display_batch_with_predictions(train_dataset, train_filenames, model, class_names, dataset_name='Training Set')

#print("\nDisplaying a batch of images with predictions from the validation dataset:")
#display_batch_with_predictions(valid_dataset, valid_filenames, model, class_names, dataset_name='Validation Set')

#print("\nDisplaying a batch of images with predictions from the test dataset:")
#display_batch_with_predictions(test_dataset, test_filenames, model, class_names, dataset_name='Test Set')















print("predicting fault from image path\n\n")

# Assuming your class_labels dictionary is correctly set up
class_names = list(class_labels.keys())

def predict_fault_from_image_path(image_path, model, class_labels, img_height=100, img_width=100):
    # Extract the filename from the image path
    filename = os.path.basename(image_path)

    # Load and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = tf.expand_dims(img, 0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_labels[predicted_class_index]
    confidence = 100 * np.max(tf.nn.softmax(predictions, axis=1)[0])

    # Print the file name along with the predicted class and confidence
    print(f"File: {filename}\nPredicted class: {predicted_class_name} with confidence {confidence:.2f}%")

# Example usage
image_path = r"boo12\pcb-defect-dataset\train\images\l_light_01_missing_hole_01_2_600.jpg"

predict_fault_from_image_path(image_path, model, class_names)

image_paths = [
    r"boo12\PCB datasetK copy\PCB_DATASET\images\Missing_hole\01_missing_hole_02.jpg",
    r"boo12\PCB datasetK copy\PCB_DATASET\images\Mouse_bite\01_mouse_bite_03.jpg",
    r"boo12\PCB datasetK copy\PCB_DATASET\images\Short\01_short_02.jpg",
    r"boo12\PCB datasetK copy\PCB_DATASET\images\Short\01_short_02.jpg",
    r"boo12\PCB datasetK copy\PCB_DATASET\images\Spur\01_spur_03.jpg"


]

for image_path in image_paths:
    predict_fault_from_image_path(image_path, model, class_names)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

