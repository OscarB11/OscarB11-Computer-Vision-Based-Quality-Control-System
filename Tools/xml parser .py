import xml.etree.ElementTree as ET
import os

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract relevant information
    filename = root.find('filename').text
    class_name = root.find('object/name').text

    # Extract bounding box coordinates for each object
    bboxes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))

    return filename, class_name, bboxes

# Specify the directory containing XML files
xml_directory = "boo12/PCB_DATASET/Annotations/Missing_hole"

# Create a list to store parsed information
data = []

# Loop through each XML file in the directory
for xml_file in os.listdir(xml_directory):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_directory, xml_file)
        filename, class_name, bboxes = parse_xml(xml_path)
        data.append({"filename": filename, "class_name": class_name, "bboxes": bboxes})

# Display the parsed data
for entry in data:
    print(entry)
