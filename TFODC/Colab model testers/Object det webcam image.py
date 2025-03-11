# Runs only on tfodmain env


import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import numpy as np
import cv2

# Paths
MODEL_DIR = r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\models\my_ssd_mobnet_640x640_m3'
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'ckpt-5')
PIPELINE_CONFIG_PATH = os.path.join(MODEL_DIR, 'pipeline.config')
LABEL_MAP_PATH = r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Tensorflow\workspace\annotations\label_map.pbtxt'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(CHECKPOINT_PATH).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

def run_object_detection(img):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Handle detection
    label_id_offset = 1
    img_with_detections = img.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        img_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.20,
        agnostic_mode=False)

    return img_with_detections

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    message = ""
    message_display_time = 4 # Seconds to display the message
    message_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the message if it's recent enough
        if message and (cv2.getTickCount() - message_timestamp) / cv2.getTickFrequency() < message_display_time:
            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Feed', frame)

        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to quit
        if key == ord('q'):
            break

        # Press 'c' to capture image and run object detection
        if key == ord('c'):
            detected_image = run_object_detection(frame)
            message = "Image captured and processed!"
            message_timestamp = cv2.getTickCount()
            
            # Show the detected image for a short duration or until key press
            while True:
                cv2.imshow('Object Detection', detected_image)
                if cv2.waitKey(1) & 0xFF == ord('c') or (cv2.getTickCount() - message_timestamp) / cv2.getTickFrequency() > message_display_time:
                    cv2.destroyWindow('Object Detection')
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
