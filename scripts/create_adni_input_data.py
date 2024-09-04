import os
import shutil

# Base directories
data_dir = "/home/groups/dlmrimnd/jacob/data"
rigid_input_dir = os.path.join(data_dir, "warped_full_head_to_sub")
fastsurfer_input_dir = os.path.join(data_dir, "fastsurfer_input_full_head")

# Function to create directory and copy file if it doesn't already exist
def copy_file_if_not_exists(src_file, dest_file):
    if not os.path.exists(dest_file):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")
    else:
        print(f"File {dest_file} already exists. Skipping.")

# Iterate over each subject directory in rigid_input_to_sub
for subject_dir in os.listdir(rigid_input_dir):
    full_subject_path = os.path.join(rigid_input_dir, subject_dir)
    
    if os.path.isdir(full_subject_path):
        subject_id = subject_dir.split('_')[0]  # Extract subject ID (e.g., sub-003S6067)
        session_ids = subject_dir.split('_')[1:]  # Extract session IDs (e.g., ses-01 and ses-02)

        # Construct paths for source and target files
        

        if len(session_ids) == 2:
            
            session1_id = session_ids[0]
            session2_id = session_ids[1]
            source_file = os.path.join(full_subject_path, f"warped_source_{session1_id}.nii.gz")
            target_file = os.path.join(full_subject_path, f"warped_target_{session2_id}.nii.gz")
            # Define the target file paths
            anat_dir1 = os.path.join(fastsurfer_input_dir, subject_id, session1_id, "anat")
            anat_dir2 = os.path.join(fastsurfer_input_dir, subject_id, session2_id, "anat")

            output_source_file = os.path.join(anat_dir1, f"{subject_id}_{session1_id}_T1w.nii.gz")
            output_target_file = os.path.join(anat_dir2, f"{subject_id}_{session2_id}_T1w.nii.gz")

            # Copy the source and target files
            copy_file_if_not_exists(source_file, output_source_file)
            copy_file_if_not_exists(target_file, output_target_file)

