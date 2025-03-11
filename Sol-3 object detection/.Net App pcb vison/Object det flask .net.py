from flask import Flask, request, jsonify
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import numpy as np
import cv2
import base64

app = Flask(__name__)

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

@app.route('/detect', methods=['POST'])
def detect():
    # Get the threshold from query parameters
    threshold = float(request.args.get('threshold', 0.3))

    # Get image from request
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

    # Create a blank image for visualization
    img_with_detections = img.copy()

    # Initialize fault counts dictionary
    fault_counts = {}

    # Filter detections based on threshold
    filtered_indices = np.where(detections['detection_scores'] >= threshold)[0]
    filtered_boxes = detections['detection_boxes'][filtered_indices]
    filtered_classes = detections['detection_classes'][filtered_indices]
    filtered_scores = detections['detection_scores'][filtered_indices]

    # Visualize detections
    viz_utils.visualize_boxes_and_labels_on_image_array(
        img_with_detections,
        filtered_boxes,
        filtered_classes + 1,  # add 1 to class id because label map ids start from 1
        filtered_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=threshold,
        agnostic_mode=False)

    # Calculate fault counts based on filtered detections
    for cls in filtered_classes:
        cls_name = category_index[cls + 1]['name']
        if cls_name not in fault_counts:
            fault_counts[cls_name] = 0
        fault_counts[cls_name] += 1

    # Encode image with detections
    _, img_encoded = cv2.imencode('.jpg', img_with_detections)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    response = {
        'image': img_base64,
        'faultCounts': fault_counts
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
