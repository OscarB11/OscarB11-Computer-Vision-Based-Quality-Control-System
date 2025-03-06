import os

def count_files_in_subfolders(root_dir):
    subfolders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    file_counts = {}
    
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        file_counts[folder] = len(os.listdir(folder_path))
        
    return file_counts

def calculate_data_split(train_count, valid_count):
    total_count = train_count + valid_count
    train_split = (train_count / total_count) * 100
    valid_split = (valid_count / total_count) * 100
    return train_split, valid_split



dataset_dir = "boo12\PCB Defects.v1i.tensorflow"
subfolder_counts = count_files_in_subfolders(dataset_dir)

train_count = subfolder_counts.get('train', 0)
valid_count = subfolder_counts.get('valid', 0)

print("Number of files/images in each subfolder:")
for folder, count in subfolder_counts.items():
    print(f"{folder}: {count} files/images")
    
print(train_count, valid_count)

train_split, valid_split = calculate_data_split(train_count, valid_count)
print("\nData Split:")
print(f"Training Data: {train_split:.2f}%")
print(f"Validation Data: {valid_split:.2f}%")



import os

def display_directory_layout_and_count_images(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}  # Add or remove extensions as needed
    print("Directory Layout with Image Counts:")
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        # Count files with image extensions
        image_count = sum(1 for f in files if os.path.splitext(f)[1].lower() in image_extensions)
        print(f"{indent}{os.path.basename(root)}/ ({image_count} images)")

# Specify the path to the folder you want to inspect
folder_path = "boo12/pcb-defect-dataset".replace('/', os.sep)  # Ensure correct path separator is used
folder_path2 = "boo12\PCB Defects.v1i.tensorflow".replace('\\', os.sep)  # Ensure correct path separator is used
folder_path3 = "boo12\PCB Labelled Images Dataset".replace('\\', os.sep)  # Ensure correct path separator is used
folder_path4 = "boo12\PCB_Dataset.v3i.voc".replace('\\', os.sep)  # Ensure correct path separator is used
folder_path5 = "boo12\PCB Defects Processed High Resolution".replace('\\', os.sep)  # Ensure correct path separator is used
folder_path6 = "boo12\PCB Defects.v1i.tensorflow Original_Format".replace('\\', os.sep)  # Ensure correct path separator is used


# Call the function with the specified folder_path
display_directory_layout_and_count_images(folder_path)
print("\n\n")
display_directory_layout_and_count_images(folder_path2)
print("\n\n")
display_directory_layout_and_count_images(folder_path3)
print("\n\n")
display_directory_layout_and_count_images(folder_path4)
print("\n\n")
display_directory_layout_and_count_images(folder_path5)
print("\n\n")
display_directory_layout_and_count_images(folder_path6)
#code to change split of images 

"""  
import os
import random
import shutil

def rearrange_for_split(root_dir, train_percent):
    # Create validation directory if it doesn't exist
    valid_dir = os.path.join(root_dir, 'valid')
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    subfolders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
    
    for folder in subfolders:
        train_folder = os.path.join(root_dir, 'train', folder)
        valid_folder = os.path.join(root_dir, 'valid', folder)

        # Create subfolder in validation directory
        if not os.path.exists(valid_folder):
            os.makedirs(valid_folder)

        files = os.listdir(train_folder)
        random.shuffle(files)  # Shuffle files to ensure randomness

        # Determine the number of files to move to the validation folder
        num_files_to_move = int(len(files) * (1 - train_percent))

        # Move files to validation folder
        for i in range(num_files_to_move):
            file_to_move = files[i]
            src = os.path.join(train_folder, file_to_move)
            dst = os.path.join(valid_folder, file_to_move)
            shutil.move(src, dst)

# Define the root directory of your dataset
dataset_dir = "PCB Defects.v1i.tensorflow"

# Rearrange images for 80-20 split (80% training, 20% validation)
rearrange_for_split(dataset_dir, train_percent=0.8)

print("Images rearranged for 80-20 split between training and validation datasets.")


"""