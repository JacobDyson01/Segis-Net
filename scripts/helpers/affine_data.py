import os
import shutil

# Define the directories
input_images_dir = '/home/groups/dlmrimnd/jacob/data/input_images'
input_affine_path = '/home/groups/dlmrimnd/jacob/data/input_affine_path'

# Create the output directory if it doesn't exist
os.makedirs(input_affine_path, exist_ok=True)

# Iterate over each subject-session directory in input_images
for subj_ses_dir in os.listdir(input_images_dir):
    subj_ses_path = os.path.join(input_images_dir, subj_ses_dir)
    
    if os.path.isdir(subj_ses_path):
        # Define the path to the warped_FA.nii.gz file
        warped_file_path = os.path.join(subj_ses_path, 'warped_FA.nii.gz')
        
        if os.path.exists(warped_file_path):
            # Create the corresponding directory in input_affine_path
            dest_subj_ses_path = os.path.join(input_affine_path, subj_ses_dir)
            os.makedirs(dest_subj_ses_path, exist_ok=True)
            
            # Move the warped_FA.nii.gz file
            shutil.move(warped_file_path, os.path.join(dest_subj_ses_path, 'warped_FA.nii.gz'))

print("Files have been moved successfully.")