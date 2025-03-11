import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time

# Path to test images
IMAGE_DIR = r"c:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\Final solution 3 object detction programs\Test images"
# Path to save output images
OUTPUT_DIR = r"c:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\Final solution 3 object detction programs\Output images"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pre-trained model
print("Loading model...")
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(model_url)
print("Model loaded successfully!")

# List of category labels for COCO dataset
category_index = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

def load_image(image_path):
    """Load and preprocess an image for model input"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    return img

def detect_objects(image_path):
    """Run object detection on a single image"""
    print(f"Processing image: {os.path.basename(image_path)}")
    
    # Load image
    original_img = Image.open(image_path)
    img_tensor = load_image(image_path)
    
    # Add batch dimension
    input_tensor = tf.expand_dims(img_tensor, 0)
    
    # Run detection
    start_time = time.time()
    detections = detector(input_tensor)
    end_time = time.time()
    
    # Convert to numpy arrays
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()
    
    # Get image dimensions
    img_width, img_height = original_img.size
    
    # Create a new image for drawing
    draw_img = original_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # Set font for labels (try to use a TrueType font if available)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw bounding boxes and labels
    for i in range(min(10, len(detection_scores))):  # Display top 10 detections
        if detection_scores[i] >= 0.3:  # Confidence threshold
            # Get coordinates
            ymin, xmin, ymax, xmax = detection_boxes[i]
            left, top, right, bottom = (xmin * img_width, ymin * img_height, 
                                      xmax * img_width, ymax * img_height)
            
            # Draw rectangle
            draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
            
            # Get class name and score
            class_id = detection_classes[i]
            if class_id in category_index:
                class_name = category_index[class_id]
                score = detection_scores[i]
                
                # Draw label
                label = f"{class_name}: {score:.2f}"
                text_width, text_height = draw.textbbox((0,0), label, font=font)[2:4]
                draw.rectangle([(left, top), (left + text_width, top + text_height)], fill="red")
                draw.text((left, top), label, fill="white", font=font)
    
    # Add processing time
    processing_time = end_time - start_time
    time_text = f"Processing time: {processing_time:.2f}s"
    draw.text((10, 10), time_text, fill="red", font=font)
    
    # Save result
    output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(image_path)}")
    draw_img.save(output_path)
    print(f"Saved result to {output_path}")
    
    return draw_img, detection_classes, detection_scores

def main():
    # Check if image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory {IMAGE_DIR} does not exist")
        return
    
    # Get all image files in directory
    image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        try:
            result_img, _, _ = detect_objects(image_path)
            
            # Display result (optional)
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(result_img))
            plt.axis('off')
            plt.title(f"Detection results: {os.path.basename(image_path)}")
            plt.show()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    print("Starting object detection program...")
    main()
    print("Object detection completed!")
