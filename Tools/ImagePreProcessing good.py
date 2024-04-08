import cv2
import numpy as np
import os

# Define the input and output directories
input_dir = "boo12\Test photos"
output_dir = "Cropped photos"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the input images
for filename in os.listdir(input_dir):
    # Read the image as a numpy array
    img = cv2.imread(os.path.join(input_dir, filename))

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask using the inRange function to isolate the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply the mask to the original image
    green_pcb = cv2.bitwise_and(img, img, mask=mask)

    # Convert the result to grayscale
    gray = cv2.cvtColor(green_pcb, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(gray, 30, 150)

    # Dilate the edges to connect nearby components
    dilated_edges = cv2.dilate(edges, None, iterations=1)

    # Find contours on the dilated edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the maximum area
        max_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon (reduce number of points)
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)

        # Find the bounding rectangle of the approximated contour
        x, y, w, h = cv2.boundingRect(approx_polygon)

        # Add a buffer/margin to avoid cropping too much
        buffer = 10
        x = max(0, x - buffer)
        y = max(0, y - buffer)
        w += 2 * buffer
        h += 2 * buffer

        # Crop the image using the bounding rectangle with added buffer
        cropped = img[y:y + h, x:x + w]

        # Save the cropped image to the output directory
        cv2.imwrite(os.path.join(output_dir, filename), cropped)

        print("done")
    else:
        print("No valid contours found in", filename)
