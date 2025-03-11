# Can run on tf_env_save 


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk



# Use the provided paths
model_path = r"C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\Final solution 2 Image classifcation programs\pcb_defects_model.h7"
image_dir = r"C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\Final solution 2 Image classifcation programs\Faulty pcb Test images"

# Define class names
class_names = [
    "missing_hole", 
    "mouse_bite", 
    "open_circuit", 
    "short", 
    "spur", 
    "spurious_copper"
]

def load_and_predict_image(model, image_path, image_size=(256, 256)):
    """
    Loads a saved model and makes a prediction on a single image.
    
    :param model: Loaded model.
    :param image_path: Path to the image file.
    :param image_size: Size to which the image will be resized (default is (256, 256)).
    :return: Prediction, class name, confidence level
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image to [0, 1]
    
    # Make a prediction
    prediction = model.predict(image)
    
    # Get the predicted class and confidence level
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence_level = np.max(prediction, axis=1)[0]
    class_name = class_names[predicted_class]
    
    return prediction, class_name, confidence_level

class PCBDefectsApp:
    def __init__(self, root, model_path, image_dir):
        self.root = root
        self.root.title("PCB Defects Classifier")
        self.model = tf.keras.models.load_model(model_path)
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.current_image_index = 0
        self.setup_ui()
        self.show_image()

    def setup_ui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Change Directory", command=self.change_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Frame for navigation and prediction buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Previous button
        self.prev_button = ttk.Button(button_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.grid(row=0, column=0, padx=5)

        # Next button
        self.next_button = ttk.Button(button_frame, text="Next", command=self.show_next_image)
        self.next_button.grid(row=0, column=1, padx=5)

        # Predict button
        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.predict_image)
        self.predict_button.grid(row=0, column=2, padx=5)

        # Change Directory button
        self.change_dir_button = ttk.Button(button_frame, text="Change Directory", command=self.change_directory)
        self.change_dir_button.grid(row=0, column=3, padx=5)
        
        # Test Individual Image button
        self.test_image_button = ttk.Button(button_frame, text="Test Individual Image", command=self.test_individual_image)
        self.test_image_button.grid(row=0, column=4, padx=5)

        # Image display
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=10)

        # Result display
        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        # Filename display
        self.filename_label = ttk.Label(self.root, text="", font=("Helvetica", 12))
        self.filename_label.pack(pady=5)
        
    def change_directory(self):
        new_dir = filedialog.askdirectory(initialdir=self.image_dir, title="Select Image Directory")
        if new_dir:
            self.image_dir = new_dir
            self.image_files = [f for f in os.listdir(new_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
            self.current_image_index = 0
            self.show_image()

    def show_image(self):
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return
        
        image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
        image = Image.open(image_path)
        image = image.resize((256, 256), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo)
        self.image_label.image = photo
        self.result_label.config(text="")
        self.filename_label.config(text=os.path.basename(image_path))

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()
    
    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image()
    
    def predict_image(self):
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return
        
        image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
        _, class_name, confidence_level = load_and_predict_image(self.model, image_path)
        self.result_label.config(text=f"Prediction: {class_name}, Confidence level: {confidence_level:.2f}")

    def test_individual_image(self):
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            _, class_name, confidence_level = load_and_predict_image(self.model, image_path)
            image = Image.open(image_path)
            image = image.resize((256, 256), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.result_label.config(text=f"Prediction: {class_name}, Confidence level: {confidence_level:.2f}")
            self.filename_label.config(text=os.path.basename(image_path))

if __name__ == "__main__":
    root = tk.Tk()
    app = PCBDefectsApp(root, model_path, image_dir)
    root.mainloop()
