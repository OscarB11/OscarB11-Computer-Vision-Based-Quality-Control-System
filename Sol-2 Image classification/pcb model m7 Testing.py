import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_and_predict(model_path, base_dir, image_size, batch_size):
    """
    Loads a saved model, allows manual navigation through predictions, and then evaluates
    the model's accuracy on the entire dataset, printing out predictions for each image.
    
    :param model_path: Path to the saved model.
    :param base_dir: Base directory containing the 'test' dataset.
    :param image_size: Size to which images are resized.
    :param batch_size: Number of images to process at once.
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Prepare data generator for the prediction dataset
    predict_datagen = ImageDataGenerator(rescale=1./255)
    predict_generator = predict_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=image_size,
        batch_size=1,  # Process one image at a time for manual navigation
        class_mode='categorical',
        shuffle=False  # Ensure order for navigation and evaluation
    )
    
    if len(predict_generator.class_indices) != 6:
        print("Error: The number of classes in the dataset does not match the model's expected number of classes (6).")
        return

    total_images = predict_generator.samples
    print(f"Total images found: {total_images}")
    
    predictions = model.predict(predict_generator, steps=total_images)
    
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_levels = np.max(predictions, axis=1)
    class_labels = list(predict_generator.class_indices.keys())
    true_classes = predict_generator.classes

    # Initialize navigation
    current_index = 0
    
    while True:
        img_path = predict_generator.filepaths[current_index]
        predicted_label = class_labels[predicted_classes[current_index]]
        confidence = confidence_levels[current_index]

        img = plt.imread(img_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"File: {os.path.basename(img_path)}\nPredicted: {predicted_label} with confidence {confidence:.2f}")
        plt.axis('off')
        plt.show()

        # Navigation input
        action = input("Enter 'n' to go to the next image, 'p' to go to the previous image, or 'q' to quit: ").strip().lower()
        if action == 'n':
            current_index = (current_index + 1) % total_images
        elif action == 'p':
            current_index = (current_index - 1) % total_images
        elif action == 'q':
            break
        else:
            print("Invalid input. Please try again.")

    print("\nModel Testing and Prediction Details:")
    
    predict_generator.reset()  # Resetting generator is crucial before detailed testing

    actual_counts = {class_label: 0 for class_label in class_labels}
    detected_counts = {class_label: 0 for class_label in class_labels}
    errors = {class_label: 0 for class_label in class_labels}

    for i in range(total_images):
        batch = predict_generator.next()  # Get next batch, which contains a single image since batch_size=1
        prediction = model.predict_on_batch(batch[0])
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]  # Extract the confidence (max prediction probability)
        predicted_class = list(predict_generator.class_indices.keys())[predicted_class_index]
        true_class = batch[1].argmax(axis=1)[0]  # True class label index
        true_class_label = list(predict_generator.class_indices.keys())[true_class]
        file_path = predict_generator.filepaths[i]
        file_name = os.path.basename(file_path)
        
        actual_counts[true_class_label] += 1
        detected_counts[predicted_class] += 1

        if predicted_class != true_class_label:
            errors[true_class_label] += 1

        print(f"File: {file_name}, True class: {true_class_label}, Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
    
    error_rates = {class_label: (errors[class_label] / actual_counts[class_label]) * 100 for class_label in class_labels}

    # Print summary table
    summary_data = {
        'Class': class_labels,
        'Actual Number': [actual_counts[class_label] for class_label in class_labels],
        'Detected Number': [f"{detected_counts[class_label]} ({'+' if errors[class_label] > 0 else ''}{errors[class_label]} error)" if errors[class_label] > 0 else detected_counts[class_label] for class_label in class_labels],
        'Error Rate (%)': [f"{error_rates[class_label]:.2f}%" for class_label in class_labels]
    }
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(summary_df)

    # After displaying predictions, reset for final evaluation
    predict_generator.reset()  
    eval_loss, eval_accuracy = model.evaluate(predict_generator, steps=total_images)
    print(f"\nTest Loss: {eval_loss:.4f}, Test Accuracy: {eval_accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Define parameters and call the function
model_path = os.path.join("boo12",'Final solution 2 Image classifcation programs', 'pcb_defects_model.h7')
base_dir = os.path.join('boo12', 'Final solution 2 Image classifcation programs','PCB Defects Processed High Resolution')
base_dir2 = os.path.join('boo12', 'PCB Defects.v1i.tensorflow Original_Format')
image_size = (256, 256)
batch_size = 1  # Set to 1 for individual image processing and navigation

load_and_predict(model_path, base_dir, image_size, batch_size)
#load_and_predict(model_path, base_dir2, image_size, batch_size)
