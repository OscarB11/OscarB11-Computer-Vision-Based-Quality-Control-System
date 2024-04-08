import os
import pandas as pd
import shutil

# Define the path to your dataset
dataset_path = "boo12\PCB Defects.v1i.tensorflow"

# Define class labels
class_labels = {
    "open_circuit": 0,
    "missing_hole": 1,
    "spurious_copper": 2,
    "short": 3,
    "mouse_bite": 4,
    "spur": 5,
    "no_defect": 6
}

# Process each directory (test, train, valid)
for folder in ["test", "train", "valid"]:
    print(f"Processing folder: {folder}")
    folder_path = os.path.join(dataset_path, folder)
    annotations_path = os.path.join(folder_path, "_annotations.csv")

    # Check if annotations file exists
    if not os.path.isfile(annotations_path):
        print(f"Annotations file missing in {folder} folder. Skipping.")
        continue

    # Read the annotations file
    print(f"Reading annotations from {annotations_path}")
    annotations = pd.read_csv(annotations_path)

    # For each class label, create a subdirectory if it doesn't exist
    for class_name in class_labels.keys():
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Created directory: {class_dir}")
        else:
            print(f"Directory already exists: {class_dir}")

    # Move each file to its class subdirectory based on the annotation
    for index, row in annotations.iterrows():
        original_file_path = os.path.join(folder_path, row['filename'])
        if not os.path.isfile(original_file_path):
            print(f"File {row['filename']} not found in {folder}. Skipping.")
            continue

        target_dir = os.path.join(folder_path, row['class'])
        target_file_path = os.path.join(target_dir, row['filename'])
        
        if not os.path.exists(target_file_path):
            shutil.move(original_file_path, target_file_path)
            print(f"Moved {row['filename']} to {target_dir}")
        else:
            print(f"File {row['filename']} already exists in target directory. Skipping.")

    print(f"Finished processing folder: {folder}")

print("Dataset restructuring complete.")
