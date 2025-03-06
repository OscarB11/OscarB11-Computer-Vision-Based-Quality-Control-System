import os
import shutil

# Define the base path where the directories are located
base_path = "boo12/PCB_Dataset.v3i.voc"

# Define paths to the test and valid directories
test_dir = os.path.join(base_path, "test")
valid_dir = os.path.join(base_path, "valid")

# Check if the directories exist
if not os.path.exists(test_dir) or not os.path.exists(valid_dir):
    print("Error: One or both directories do not exist.")
else:
    # Get the list of files in the valid directory
    valid_files = os.listdir(valid_dir)
    total_files = len(valid_files)
    moved_count = 0
    skipped_count = 0

    print(f"Starting to move files from '{valid_dir}' to '{test_dir}'.")

    # Move all files from the valid directory to the test directory
    for index, filename in enumerate(valid_files):
        src_file = os.path.join(valid_dir, filename)
        dst_file = os.path.join(test_dir, filename)

        # Check if the file already exists in the test directory
        if os.path.exists(dst_file):
            print(f"Skipping {filename} (already exists in test).")
            skipped_count += 1
        else:
            shutil.move(src_file, dst_file)
            print(f"Moved {filename} from valid to test ({index + 1}/{total_files}).")
            moved_count += 1

    # Optionally, remove the valid directory if empty
    if not os.listdir(valid_dir):
        os.rmdir(valid_dir)
        print("Valid directory is empty and has been removed.")

    print(f"Moving complete: {moved_count} files moved, {skipped_count} files skipped.")
