import os

# Define the directory path
warped_input_dir = '/home/groups/dlmrimnd/jacob/data/MND/input_images'

# Iterate through the subfolders in the warped_input_roi directory
for folder_name in os.listdir(warped_input_dir):
    folder_path = os.path.join(warped_input_dir, folder_name)
    
    # Check if it is a directory and has the old format (_ instead of -)
    if os.path.isdir(folder_path) and '_ses_' in folder_name:
        # Replace underscores with hyphens for session identifiers
        new_folder_name = folder_name.replace('_ses_', '_ses-')
        
        # Construct the full new folder path
        new_folder_path = os.path.join(warped_input_dir, new_folder_name)
        
        # Rename the folder
        os.rename(folder_path, new_folder_path)
        
        print(f"Renamed: {folder_name} -> {new_folder_name}")

print("Renaming complete!")
