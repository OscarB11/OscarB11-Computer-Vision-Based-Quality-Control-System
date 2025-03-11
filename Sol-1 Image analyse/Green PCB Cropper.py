import cv2
import numpy as np
import os

# Define the input and output directories
input_dir = "boo12\Sol-1 image analyse\Cropping Test photos"
output_dir = "boo12\Sol-1 image analyse\Cropped photos"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Loop through the input images
for filename in os.listdir(input_dir):
    # Read the image as a numpy array
    img = cv2.imread(os.path.join(input_dir, filename))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 5)
    
    # Dilate thresholded image to connect all parts of PCB 
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    # This step helps to avoid gaps or holes in the PCB contour
    
    # Find contours on dilated image 
    contours, hierarchy = cv2.findContours(dilation,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # This step finds the outermost contours of the PCB
    
    # Find the largest contour, assuming it is the PCB
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    # Find the bounding rectangle of the PCB
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Add a buffer/margin to avoid cropping too much
    buffer = 10  # Adjust this value as needed
    x = max(0, x - buffer)
    y = max(0, y - buffer)
    w += 2 * buffer
    h += 2 * buffer
    # This step adds some extra space around the PCB contour
    
    # Crop the image using the bounding rectangle with added buffer
    cropped = img[y:y+h, x:x+w]
    
    # Save the cropped image to output directory
    cv2.imwrite(os.path.join(output_dir, filename), cropped)
    
    print("done")
