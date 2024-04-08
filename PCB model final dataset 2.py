from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="azRWiGEfbIy7c0ul9hCN"
)


result = CLIENT.infer(
    r"boo12\Screenshot 2024-04-07 031749.png",
    model_id="pcb-defect-94yfu/1"
)


print(result)
image_path = r"boo12\Screenshot 2024-04-07 031749.png"
image = cv2.imread(image_path)

# Ensure the image was loaded
if image is None:
    print("Error: Image not found. Check the file path.")
else:
    # Prediction result (simplified for this example)
    predictions = [
        {
            "x": 118.5,
            "y": 360.5,
            "width": 35.0,
            "height": 29.0,
            "confidence": 0.5062018632888794,
            "class": "missing_hole",
            "class_id": 0
        }
    ]

    # Draw each prediction on the image
    for pred in predictions:
        # Calculate the bounding box coordinates
        start_point = (int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2))
        end_point = (int(pred["x"] + pred["width"] / 2), int(pred["y"] + pred["height"] / 2))

        # Draw the bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

        # Put the class name and confidence on the image
        text = f"{pred['class']}: {pred['confidence']:.2f}"
        cv2.putText(image, text, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Display the image
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()
