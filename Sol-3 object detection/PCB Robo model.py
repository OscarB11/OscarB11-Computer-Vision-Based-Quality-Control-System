from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

# Initialize the Inference HTTP Client with your API details
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="azRWiGEfbIy7c0ul9hCN"
)

# Define a function to overlay detections on an image
def annotate_image(image, detections):
    """
    Annotate the image with detection results.
    
    Args:
    - image: The image to annotate.
    - detections: A list of detections, where each detection is a dictionary containing 'x', 'y', 'width', 'height', 'confidence', 'class'.
    """
    for detection in detections:
        # Calculate the bounding box coordinates
        start_point = (int(detection["x"] - detection["width"] / 2), int(detection["y"] - detection["height"] / 2))
        end_point = (int(detection["x"] + detection["width"] / 2), int(detection["y"] + detection["height"] / 2))
        
        # Draw the bounding box
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        
        # Put the class name and confidence on the image
        text = f"{detection['class']}: {detection['confidence']:.2f}"
        cv2.putText(image, text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# List of image paths
image_paths =[
    r"C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\Final solution 3 object detction programs\Screenshot 2024-04-07 031749.png",
    r"C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\TFODC\Colab model testers\Colab model Test image\08_missing_hole_02.jpg"
]

# Iterate through each image, infer detections, and display results
for image_path in image_paths:
    # Infer defects using the client
    result = CLIENT.infer(image_path, model_id="pcb-defect-94yfu/1")

    # Load the image
    image = cv2.imread(image_path)
    if image is None: 
        print(f"Error: Image not found at {image_path}. Check the file path.")
        continue
    
    # Assuming the result format matches the simplified example provided earlier
    # If your result format is different, adjust the parsing accordingly
    detections = result['predictions']  # This line may need adjustment based on your actual result format
    
    # Annotate the image with detections
    annotate_image(image, detections)
    
    # Display the image with annotations
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




