

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from itertools import islice

# Define the base directory for the processed dataset
base_dir = os.path.join('boo12', 'PCB Defects Processed High Resolution')

# Define image size and batch size
image_size = (256, 256)
batch_size = 32#

# Prepare ImageDataGenerator and data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # using 20% of the training data for validation
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'valid'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 35
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)


# Evaluate the model on the test data
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Calculate the predictions on the test data
test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict(test_generator, steps=test_steps_per_epoch)
# Get most likely class indices
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix')
print(conf_matrix)

# Classification report
print('Classification Report')
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Select 15 random images from the test set
num_random_samples = 15
random_indices = np.random.choice(np.arange(len(test_generator.filenames)), num_random_samples, replace=False)
sample_filenames = [test_generator.filenames[idx] for idx in random_indices]
sample_images = [test_generator.filepaths[idx] for idx in random_indices]
true_labels = [class_labels[true_classes[idx]] for idx in random_indices]
predicted_labels = [class_labels[predicted_classes[idx]] for idx in random_indices]
confidences = [predictions[idx][predicted_classes[idx]] for idx in random_indices]

# Display the 15 random images with their true labels, predicted labels, and confidence
plt.figure(figsize=(15, 10))
for i, (filename, true_label, predicted_label, confidence) in enumerate(zip(sample_filenames, true_labels, predicted_labels, confidences), 1):
    img = plt.imread(sample_images[i - 1])
    plt.subplot(3, 5, i)
    plt.imshow(img)
    plt.title(f"Filename: {filename}\nTrue: {true_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.2f}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# Save the model
model.save(os.path.join("boo12", 'pcb_defects_model.h7'))


