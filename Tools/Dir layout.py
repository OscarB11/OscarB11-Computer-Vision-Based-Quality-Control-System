import os

def display_directory_layout(folder_path):
    print("Directory Layout:")
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")

# Specify the path to the folder you want to inspect
folder_path = "boo12/pcb-defect-dataset"
folder_path2 = "boo12\PCB Defects.v1i.tensorflow"

# Call the function with the specified folder_path
display_directory_layout(folder_path)
print("\n\n")
display_directory_layout(folder_path2)
