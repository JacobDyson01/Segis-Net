import os
import nibabel as nib
import numpy as np

# Directories to iterate through
warped_input_dir = "/home/groups/dlmrimnd/jacob/data/MND/warped_input_roi"
output_segmentation_dir = "/home/groups/dlmrimnd/jacob/data/MND/output_segmentations"
etiv_output_dir = "/home/groups/dlmrimnd/akshit/MND_output_files"
save_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_MND"

# Function to extract eTIV from the aseg.stats file
def extract_etiv_from_aseg(file_path):
    etiv = None
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the eTIV information
            if "EstimatedTotalIntraCranialVol" in line:
                etiv = float(line.split(",")[-2])  # Extract eTIV value from the line
                break
    return etiv

# Function to calculate volume from the segmentation NIfTI file
def calculate_volume_from_segmentation(segmentation_file):
    segmentation_img = nib.load(segmentation_file)
    segmentation_data = segmentation_img.get_fdata()

    # Assuming segmentation values are binary (1 = ROI, 0 = background)
    roi_voxel_count = np.sum(segmentation_data == 1)

    # Voxel size is 1mm x 1mm x 1mm, so volume in mm³ equals the number of voxels
    volume_mm3 = roi_voxel_count
    return volume_mm3

# Iterate through each subject folder in the warped_input_dir
for subject_folder in os.listdir(warped_input_dir):
    
    subject_path = os.path.join(warped_input_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        # Split the session names to find the first session (ses-01)
        sessions = subject_folder.split('_')
        first_session = sessions[1]  # Assuming format is sub-xxx_ses-01_ses-02
        
        # Construct the subject ID and first session directory for the eTIV
        subject_id = sessions[0]
        first_session_dir = f"{subject_id}_{first_session}"

        # Locate the aseg.stats file in the corresponding folder
        aseg_stats_file_path = os.path.join(etiv_output_dir, first_session_dir, 'stats', 'aseg.stats')
        
        if os.path.exists(aseg_stats_file_path):
            # Extract the eTIV for the first session
            etiv = extract_etiv_from_aseg(aseg_stats_file_path)

            if etiv is not None and etiv > 0:
                # Find the corresponding target segmentation NIfTI file
                target_segmentation_file = os.path.join(output_segmentation_dir, subject_folder, 'target_segmentation.nii.gz')
                
                if os.path.exists(target_segmentation_file):
                    # Calculate the volume of the ROI from the segmentation
                    total_volume = calculate_volume_from_segmentation(target_segmentation_file)
                    
                    # Calculate the percentage of volume relative to eTIV
                    volume_percent = (total_volume / etiv) * 100

                    # Create the output directory for volumes if it doesn't exist
                    subject_volume_dir = os.path.join(save_dir, subject_folder)
                    os.makedirs(subject_volume_dir, exist_ok=True)
                    
                    # Save the percentage volume to a file
                    volume_file_path = os.path.join(subject_volume_dir, 'volume.txt')
                    with open(volume_file_path, 'w') as volume_file:
                        volume_file.write(f"Volume as percentage of eTIV: {volume_percent:.4f}%\n")
                    
                    print(f"Saved percentage volume for {subject_folder} to {volume_file_path}")
                else:
                    print(f"Segmentation file not found for {subject_folder}")
            else:
                print(f"eTIV value not found or invalid for {subject_folder}")
        else:
            print(f"aseg.stats file not found for {first_session_dir}")
