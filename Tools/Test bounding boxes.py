import os
import csv
from PIL import Image, ImageDraw

# Adjust the base directory using a raw string for the initial part of the path
# and os.path.join for the rest to ensure compatibility across operating systems
base_dir = os.path.join(r'boo12\PCB Defects.v1i.tensorflow', 'test')
annotations_path = os.path.join(base_dir, '_annotations.csv')

# Function to draw bounding boxes on images
def draw_boxes(image_path, boxes):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box, outline='red', width=2)
        
        # Ensure the directory exists before saving
        output_dir = os.path.join(os.path.dirname(image_path), 'test_annotated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to a new directory
        img.save(os.path.join(output_dir, os.path.basename(image_path)))

# Read annotations and process images
with open(annotations_path, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    current_file = None
    boxes = []
    for row in reader:
        # Adjust for the filename path, using os.path.join and replacing backslashes if necessary
        class_dir = row['class']
        filename = row['filename'].replace('/', os.sep).replace('\\', os.sep)
        image_path = os.path.join(base_dir, class_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
        
        if current_file != filename:
            if current_file is not None:
                # Process the previous image
                draw_boxes(current_image_path, boxes)
            current_file = filename
            boxes = [(int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))]
            current_image_path = image_path  # Store current image path
        else:
            boxes.append((int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])))
            
    # Ensure the last image is processed
    if current_file is not None:
        draw_boxes(current_image_path, boxes)
