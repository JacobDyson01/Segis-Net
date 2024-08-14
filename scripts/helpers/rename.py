import os

# Define the root directory containing the input_images folder
root_dir = '/home/groups/dlmrimnd/jacob/data/input_images/'

# Iterate through each subdirectory in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    # Check if it is a directory
    if os.path.isdir(subdir_path):
        # Define the paths for the source and target files
        src_path = os.path.join(subdir_path, 'src_FA.nii.gz')
        tgt_path = os.path.join(subdir_path, 'tgt_FA.nii.gz')
        
        # Rename the files if they exist
        if os.path.exists(src_path):
            new_src_path = os.path.join(subdir_path, 'target.nii.gz')
            os.rename(src_path, new_src_path)
            print(f'Renamed {src_path} to {new_src_path}')
        
        if os.path.exists(tgt_path):
            new_tgt_path = os.path.join(subdir_path, 'source.nii.gz')
            os.rename(tgt_path, new_tgt_path)
            print(f'Renamed {tgt_path} to {new_tgt_path}')