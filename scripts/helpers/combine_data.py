import os
import shutil

# Define source and destination directories for warped_input_roi
ants_input_dir = '/home/groups/dlmrimnd/jacob/data/ants_warped_input_roi_new'
mnd_input_dir = '/home/groups/dlmrimnd/jacob/data/MND/warped_input_roi'
combined_input_dir = '/home/groups/dlmrimnd/jacob/data/combined_data/warped_input_roi'

# Define source and destination directories for warped_masks_roi
ants_masks_dir = '/home/groups/dlmrimnd/jacob/data/warped_masks_roi_new'
mnd_masks_dir = '/home/groups/dlmrimnd/jacob/data/MND/warped_binary_masks_roi'
combined_masks_dir = '/home/groups/dlmrimnd/jacob/data/combined_data/warped_masks_roi'

# Define source and destination directories for deformation_fields_roi
ants_deform_dir = '/home/groups/dlmrimnd/jacob/data/deformation_fields_roi_new'
mnd_deform_dir = '/home/groups/dlmrimnd/jacob/data/MND/deformation_fields_roi'
combined_deform_dir = '/home/groups/dlmrimnd/jacob/data/combined_data/deformation_fields_roi'

# Create the combined directories if they don't exist
os.makedirs(combined_input_dir, exist_ok=True)
os.makedirs(combined_masks_dir, exist_ok=True)
os.makedirs(combined_deform_dir, exist_ok=True)

# Function to copy subfolders from source to destination
def combine_folders(source_dir, destination_dir):
    for folder_name in os.listdir(source_dir):
        source_folder_path = os.path.join(source_dir, folder_name)
        destination_folder_path = os.path.join(destination_dir, folder_name)
        # Copy the folder if it's not already present in the destination
        if os.path.isdir(source_folder_path):
            if not os.path.exists(destination_folder_path):
                shutil.copytree(source_folder_path, destination_folder_path)
            else:
                print(f"Folder {folder_name} already exists in {destination_dir}")

# Combine the subfolders for warped_input_roi
combine_folders(ants_input_dir, combined_input_dir)
combine_folders(mnd_input_dir, combined_input_dir)

# Combine the subfolders for warped_masks_roi
combine_folders(ants_masks_dir, combined_masks_dir)
combine_folders(mnd_masks_dir, combined_masks_dir)

# Combine the subfolders for deformation_fields_roi
combine_folders(ants_deform_dir, combined_deform_dir)
combine_folders(mnd_deform_dir, combined_deform_dir)

print("Folders combined successfully!")
