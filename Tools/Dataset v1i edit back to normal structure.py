import os
import pandas as pd
import shutil

# Define the path to your original dataset and the new dataset
original_dataset_path = "boo12/PCB Defects.v1i.tensorflow"
new_dataset_path = "boo12/PCB_Defects_v1i_Original_Format"

# Ensure the new dataset directory exists
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

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
    original_folder_path = os.path.join(original_dataset_path, folder)
    new_folder_path = os.path.join(new_dataset_path, folder)
    
    # Create the folder in the new dataset directory
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    annotations_path = os.path.join(original_folder_path, "_annotations.csv")

    # Check if annotations file exists
    if not os.path.isfile(annotations_path):
        print(f"Annotations file missing in {folder} folder. Skipping.")
        continue

    # Read the annotations file
    print(f"Reading annotations from {annotations_path}")
    annotations = pd.read_csv(annotations_path)

    # For each class label, move the files back to the parent directory in the new dataset structure
    for class_name in class_labels.keys():
        class_dir = os.path.join(original_folder_path, class_name)
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                source_path = os.path.join(class_dir, filename)
                destination_path = os.path.join(new_folder_path, filename)
                
                if os.path.isfile(source_path):
                    shutil.copy(source_path, destination_path)
                    print(f"Copied {filename} from {class_dir} to {new_folder_path}")

    # Copy the annotations file to the new directory
    shutil.copy(annotations_path, new_folder_path)
    print(f"Copied annotations file to {new_folder_path}")

    print(f"Finished processing folder: {folder}")

print("New dataset creation complete.")
