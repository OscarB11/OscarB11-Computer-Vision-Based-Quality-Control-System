import os
from PIL import Image

def get_image_details(image_path):
    # Load the image
    with Image.open(image_path) as img:
        print(f"Image Format: {img.format}")
        print(f"Image Size: {img.size} pixels")  # Width and height
        print(f"Image Mode: {img.mode}")  # Color mode, e.g., RGB, RGBA, L (luminance)

        # Print DPI if available
        dpi = img.info.get('dpi')
        if dpi:
            print(f"DPI: {dpi}")
        else:
            print("DPI information not available.")

        # Access and print some metadata if available
        exif_data = img._getexif()
        if exif_data:
            print("\nAvailable EXIF Data:")
            for tag, value in exif_data.items():
                tag_name = Image.ExifTags.TAGS.get(tag, tag)
                print(f"{tag_name}: {value}")
        else:
            print("\nNo EXIF data found.")

# Specify the path to your image as a string, not a list
image_path = r'C:\Users\besto\Documents\Local vscode\CSProjectcode\boo12\PCB_Dataset.v3i.voc\test\l_light_01_missing_hole_12_3_600_jpg.rf.df723bc3dd4a2892f1c41bf8575fe7e8.jpg'
image_path = r'C:\Users\besto\Documents\Local vscode\Tensorflow Object Detection\TFODCourse\Tensorflow\workspace\images\testm1OD\04_missing_hole_03.jpg'



# Make sure the file exists
if os.path.exists(image_path):
    get_image_details(image_path)
else:
    print("File does not exist.")
