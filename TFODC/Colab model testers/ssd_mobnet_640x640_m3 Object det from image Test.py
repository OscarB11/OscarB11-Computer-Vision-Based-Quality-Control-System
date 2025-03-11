# Runs only on tfodmain env


import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import numpy as np
import cv2
import argparse

def setup_paths():
    """Configure all necessary file paths."""
    paths = {
        'MODEL_DIR': r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\models\my_ssd_mobnet_640x640_m3',
        'CHECKPOINT_PATH': r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\models\my_ssd_mobnet_640x640_m3\ckpt-5.index',
        'PIPELINE_CONFIG_PATH': r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\models\my_ssd_mobnet_640x640_m3\pipeline.config',
        'LABEL_MAP_PATH': r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\annotations\label_map.pbtxt',
        'IMAGE_DIR': r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Colab model testers\Colab model Test image'

    }
    
    # Verify paths exist
    for key, path in paths.items():
        if key != 'CHECKPOINT_PATH':  # Skip checking index file
            base_path = path.split('.')[0] if '.' in os.path.basename(path) else path
            if not os.path.exists(base_path):
                print(f"Warning: {key} path does not exist: {path}")
                
    return paths

def load_model(config_path, checkpoint_path):
    """Load and prepare the detection model."""
    try:
        # Load pipeline config and build detection model
        configs = config_util.get_configs_from_pipeline_file(config_path)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(checkpoint_path.replace('.index', '')).expect_partial()
        
        return detection_model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

@tf.function
def detect_fn(image, detection_model):
    """Run detection on an image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def process_detections(detections):
    """Process detection results into a more usable format."""
    # Convert to numpy arrays and remove batch dimension
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections

def run_object_detection(image_path, detection_model, category_index, min_score_thresh=0.15, scale_percent=50, wait_key=0):
    """Run object detection on a single image and display results."""
    # Read image from file
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    
    try:
        # Convert image to tensor
        input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)
        
        # Process detections
        detections = process_detections(detections)
        
        # Count faults by type
        label_id_offset = 1
        fault_counts = {}
        
        for i in range(detections['num_detections']):
            if detections['detection_scores'][i] >= min_score_thresh:
                cls = detections['detection_classes'][i] + label_id_offset
                if cls in category_index:
                    cls_name = category_index[cls]['name']
                    if cls_name not in fault_counts:
                        fault_counts[cls_name] = 0
                    fault_counts[cls_name] += 1
        
        # Create a copy for visualization
        img_with_detections = img.copy()
        
        # Add detection boxes and labels to the image
        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,  
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False)
        
        # Resize image for display if needed
        if scale_percent != 100:
            width = int(img_with_detections.shape[1] * scale_percent / 100)
            height = int(img_with_detections.shape[0] * scale_percent / 100)
            resized_img = cv2.resize(img_with_detections, (width, height), interpolation=cv2.INTER_AREA)
        else:
            resized_img = img_with_detections
            
        # Display output
        cv2.imshow(f'Object Detection - {os.path.basename(image_path)}', resized_img)
        
        # Print results
        print(f"\nImage: {os.path.basename(image_path)}")
        print("Detected faults:")
        if fault_counts:
            for fault_type, count in fault_counts.items():
                print(f"  {fault_type}: {count}")
        else:
            print("  None detected above threshold")
            
        # Wait for key press
        cv2.waitKey(wait_key)
        
        return fault_counts
        
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        return {}

def process_image_directory(image_dir, detection_model, category_index, min_score_thresh=0.10, scale_percent=50):
    """Process all images in a directory."""
    try:
        # Get all image files in the directory
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images to process.")
        
        # Process each image
        all_results = {}
        for i, filename in enumerate(image_files):
            print(f"\nProcessing image {i+1}/{len(image_files)}: {filename}")
            image_path = os.path.join(image_dir, filename)
            
            # Process with wait_key=0 for the last image, and 1 for others
            wait_key = 0 if i == len(image_files) - 1 else 1
            results = run_object_detection(
                image_path, detection_model, category_index, 
                min_score_thresh=min_score_thresh,
                scale_percent=scale_percent,
                wait_key=wait_key
            )
            all_results[filename] = results
        
        # Close all windows when finished
        cv2.destroyAllWindows()
        
        return all_results
    
    except Exception as e:
        print(f"Error processing directory: {str(e)}")
        return {}

def main():
    """Main function to run the object detection."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run object detection on images')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Minimum score threshold for detections (0-1)')
    parser.add_argument('--scale', type=int, default=50,
                        help='Scale percent for display (10-100)')
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_paths()
    
    # Load model
    detection_model = load_model(paths['PIPELINE_CONFIG_PATH'], paths['CHECKPOINT_PATH'])
    if detection_model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load label map
    try:
        category_index = label_map_util.create_category_index_from_labelmap(
            paths['LABEL_MAP_PATH'], use_display_name=True)
    except Exception as e:
        print(f"Error loading label map: {str(e)}")
        return
        
    # Process images
    results = process_image_directory(
        paths['IMAGE_DIR'], 
        detection_model, 
        category_index,
        min_score_thresh=args.threshold,
        scale_percent=args.scale
    )

    # Print summary
    print("\n==== Detection Summary ====")
    for image_name, counts in results.items():
        print(f"{image_name}: {counts}")

if __name__ == "__main__":
    main()
