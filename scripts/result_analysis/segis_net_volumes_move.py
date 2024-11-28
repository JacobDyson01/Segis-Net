import os
import shutil

# Source directory where the source and target segmentations are stored (e.g., sub-xxx_ses-01_ses-02 folders)
segisnet_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_MND_final_segis_net"

# New directory to save the segmentation volumes by session
final_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_final"

# Create the final directory if it doesn't exist
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

# Loop through each folder in segisnet_dir to extract both source and target volumes
for folder in sorted(os.listdir(segisnet_dir)):
    folder_path = os.path.join(segisnet_dir, folder)

    # Check if the folder is a directory and has a valid session pair (e.g., sub-xxx_ses-01_ses-02)
    if os.path.isdir(folder_path) and '_ses-' in folder:
        session_pair = folder.split('_')[-2:]  # Get the session pair (e.g., ['ses-01', 'ses-02'])
        session_start = session_pair[0]  # Keep the original session start format (e.g., ses-01)
        session_end = session_pair[1]    # Keep the original session end format (e.g., ses-02)
        
        # Construct the source and target volume file paths in the current folder
        source_volume_file = os.path.join(folder_path, 'source_volume.txt')
        target_volume_file = os.path.join(folder_path, 'target_volume.txt')
        
        # Create subject folder (e.g., sub-xxx_ses-01) in the final directory
        subject_id = '_'.join(folder.split('_')[:-2])  # Extract subject id (e.g., sub-xxx)
        
        # -- Handle Source Volume --
        # If the source volume exists, copy it as ses_XX source volume
        if os.path.exists(source_volume_file):
            # Create the directory for the session start (e.g., sub-xxx_ses-01)
            session_start_dir = os.path.join(final_dir, f"{subject_id}_{session_start}")
            os.makedirs(session_start_dir, exist_ok=True)
            
            # Copy the source volume to the new session directory as volume.txt
            new_source_volume_file = os.path.join(session_start_dir, 'volume.txt')
            shutil.copy(source_volume_file, new_source_volume_file)
            print(f"Copied source volume for {folder} to {new_source_volume_file}")
        else:
            print(f"Source volume file not found for {folder}")
        
        # -- Handle Target Volume --
        # If the target volume exists, copy it as ses_XX target volume
        if os.path.exists(target_volume_file):
            # Create the directory for the session end (e.g., sub-xxx_ses-02)
            session_end_dir = os.path.join(final_dir, f"{subject_id}_{session_end}")
            os.makedirs(session_end_dir, exist_ok=True)
            
            # Copy the target volume to the new session directory as volume.txt
            new_target_volume_file = os.path.join(session_end_dir, 'volume.txt')
            shutil.copy(target_volume_file, new_target_volume_file)
            print(f"Copied target volume for {folder} to {new_target_volume_file}")
        else:
            print(f"Target volume file not found for {folder}")

