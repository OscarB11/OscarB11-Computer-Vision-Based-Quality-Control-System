import os
import csv
from PIL import Image, ImageOps
from tqdm import tqdm  # Import tqdm for the progress bar

# Define the base directories for the original and the new datasets
original_base_dir = os.path.join('boo12', 'PCB Defects.v1i.tensorflow')
new_base_dir = os.path.join('boo12', 'PCB Defects Processed High Resolution')

# Amount of padding to add to the bounding box to include more context
padding = 15

# Higher standard size to which each fault image will be resized
standard_size = (128, 128)  # Adjust this value for your desired resolution

# Function to process an image and save the fault portion
def process_image(image_path, bbox, save_dir, filename):
    # Check if the image file exists to avoid FileNotFoundError
    if not os.path.exists(image_path):
        print(f"Cannot find image: {image_path}")
        return  # Skip this image
    
    with Image.open(image_path) as img:
        # Calculate new bounding box with padding
        padded_bbox = (
            max(bbox[0] - padding, 0),
            max(bbox[1] - padding, 0),
            min(bbox[2] + padding, img.width),
            min(bbox[3] + padding, img.height)
        )
        # Crop the image to the padded bounding box
        cropped_img = img.crop(padded_bbox)
        # Resize the cropped image to the higher standard size
        resized_img = cropped_img.resize(standard_size, Image.Resampling.LANCZOS)
        # Save the resized image to the new directory
        os.makedirs(save_dir, exist_ok=True)
        resized_img.save(os.path.join(save_dir, filename))

# Function to create the new dataset
def create_dataset(data_type):
    annotations_path = os.path.join(original_base_dir, data_type, '_annotations.csv')
    # Open the annotations CSV file and read all rows into memory
    with open(annotations_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # Load all rows into a list for the progress bar
        
    # Initialize the progress bar
    progress_bar = tqdm(total=len(rows), desc=f"Processing {data_type}")
    
    for row in rows:
        fault_class = row['class']
        filename = row['filename']
        image_path = os.path.join(original_base_dir, data_type, fault_class, filename)
        bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
        new_filename = f"processed_{filename}"
        save_dir = os.path.join(new_base_dir, data_type, fault_class)
        process_image(image_path, bbox, save_dir, new_filename)
        
        # Update the progress bar
        progress_bar.update(1)
    
    # Close the progress bar after the loop
    progress_bar.close()

# Create the new dataset for each data type
for data_type in ['train', 'valid', 'test']:
    create_dataset(data_type)

print("New high-resolution dataset created successfully.")
