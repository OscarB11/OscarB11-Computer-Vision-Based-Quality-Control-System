import os
import csv
from PIL import Image, ImageDraw
from collections import defaultdict

# Base directory for the dataset
base_dir = os.path.join(r'boo12')
# Define directories for train, valid, and test datasets
data_dirs = {
    'train': os.path.join(base_dir, 'PCB Defects.v1i.tensorflow', 'train'),
    'valid': os.path.join(base_dir, 'PCB Defects.v1i.tensorflow', 'valid'),
    'test': os.path.join(base_dir, 'PCB Defects.v1i.tensorflow', 'test'),
}
annotated_images_dir = os.path.join(base_dir, 'PCB Labelled Images Dataset')

# Initialize counters
class_counters = defaultdict(lambda: {'found': 0, 'not_found': 0})

def draw_boxes(image_path, boxes, filename, fault_class):
    # Ensure the class-specific directory exists
    class_dir = os.path.join(annotated_images_dir, fault_class)
    os.makedirs(class_dir, exist_ok=True)
    
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box, outline='red', width=2)
        
        # Save annotated image
        img.save(os.path.join(class_dir, filename))

# Process each dataset directory
for dataset_type, dataset_dir in data_dirs.items():
    annotations_path = os.path.join(dataset_dir, '_annotations.csv')

    with open(annotations_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        current_file = None
        boxes = []
        current_class = None
        for row in reader:
            fault_class = row['class']
            filename = row['filename'].replace('/', os.sep).replace('\\', os.sep)
            image_path = os.path.join(dataset_dir, fault_class, filename)
            
            if current_file != filename:
                if current_file is not None and boxes:
                    draw_boxes(current_image_path, boxes, current_file, current_class)
                    class_counters[current_class]['found'] += 1
                current_file = filename
                current_image_path = image_path  # Store current image path
                current_class = fault_class  # Update current fault class
                boxes = []  # Reset boxes for the new image
            
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                class_counters[fault_class]['not_found'] += 1
                continue
            
            # Add bounding box
            boxes.append((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))
                
        # Process the last image if needed
        if current_file is not None and boxes:
            draw_boxes(current_image_path, boxes, current_file, current_class)
            class_counters[current_class]['found'] += 1

# Print summary statistics
for fault_class, counters in class_counters.items():
    print(f"Fault Type: {fault_class}")
    print(f"  Images found and processed: {counters['found']}")
    print(f"  Images not found: {counters['not_found']}")
