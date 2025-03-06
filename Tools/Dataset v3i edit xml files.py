import os
import xml.etree.ElementTree as ET

# Define the base path where the directories are located
base_path = "boo12/PCB_Dataset.v3i.voc"

# Define paths to the test and valid directories
test_dir = os.path.join(base_path, "test")
valid_dir = os.path.join(base_path, "train")

# Logs to keep track of actions and errors
update_logs = []
error_logs = []
deleted_files = []
all_files_processed = set()  # Set to track all filenames processed across both directories

def process_directory(directory):
    # Process all XML files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            filepath = os.path.join(directory, filename)

            # Check if the file has already been processed in any directory
            if filename in all_files_processed:
                error_message = f"Duplicate file found across directories and removed: {filepath}"
                print(error_message)
                error_logs.append(error_message)
                os.remove(filepath)  # Remove the duplicate XML file
                deleted_files.append(filepath)
                image_path = filepath.replace('.xml', '.jpg')  # Assuming image extension is .jpg
                if os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_files.append(image_path)
                continue
            else:
                all_files_processed.add(filename)  # Mark this file as processed

            tree = ET.parse(filepath)
            root = tree.getroot()

            # Perform file-specific processing here (same as previously detailed)

# Process each directory
for dir_path in [test_dir, valid_dir]:
    print(f"Processing directory: {dir_path}")
    process_directory(dir_path)

print("XML processing complete.")

# Print summary report
print("\nSummary Report:")
print(f"Total files updated: {len(update_logs)}")
if update_logs:
    print("\nUpdated Files:")
    for log in update_logs:
        print(log)

print(f"Total errors encountered: {len(error_logs)}")
if error_logs:
    print("\nError Details:")
    for error in error_logs:
        print(error)

print(f"Total files deleted: {len(deleted_files)}")
if deleted_files:
    print("\nDeleted Files:")
    for file in deleted_files:
        print(file)
