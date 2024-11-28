import os
import nibabel as nib
import numpy as np

# Directories to iterate through
warped_input_dir = "/home/groups/dlmrimnd/jacob/data/MND/warped_input_roi"
output_segmentation_dir = "/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_final"
etiv_output_dir = "/home/groups/dlmrimnd/akshit/MND_output_files"
save_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_MND_final_segis_net"

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

# Function to attempt locating folder, considering potential \r issues
def locate_stats_file(folder_name, etiv_output_dir):
    # Try the folder as-is
    aseg_stats_file_path = os.path.join(etiv_output_dir, folder_name, 'stats', 'aseg.stats')

    # If the folder doesn't exist, try appending '\r' characters (based on the issue seen)
    if not os.path.exists(aseg_stats_file_path):
        folder_name_with_r = folder_name + '\r'
        aseg_stats_file_path = os.path.join(etiv_output_dir, folder_name_with_r, 'stats', 'aseg.stats')
    
    # Check again with the modified folder
    if os.path.exists(aseg_stats_file_path):
        return aseg_stats_file_path
    else:
        return None

# Iterate through each subject folder in the warped_input_dir
for subject_folder in os.listdir(warped_input_dir):
    subject_path = os.path.join(warped_input_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        # Extract the subject ID and session info
        sessions = subject_folder.split('_')
        subject_id = sessions[0]  # sub-xxx
        first_session = sessions[1]  # ses-01
        
        # Construct the directory path for the first session segmentation
        first_session_dir = f"{subject_id}_{first_session}"
        source_segmentation_file = os.path.join(output_segmentation_dir, subject_folder, 'source_segmentation.nii.gz')
        target_segmentation_file = os.path.join(output_segmentation_dir, subject_folder, 'target_segmentation.nii.gz')
        
        # Check if both the source and target segmentation files exist
        if os.path.exists(source_segmentation_file) and os.path.exists(target_segmentation_file):
            # Calculate the volume from the source and target segmentation NIfTI files
            source_volume = calculate_volume_from_segmentation(source_segmentation_file)
            target_volume = calculate_volume_from_segmentation(target_segmentation_file)
            
            # Attempt to locate the aseg.stats file in Akshit's directory for the eTIV of the first session
            aseg_stats_file_path = locate_stats_file(first_session_dir, etiv_output_dir)
            
            if aseg_stats_file_path:
                # Extract the eTIV value for the first session
                etiv = extract_etiv_from_aseg(aseg_stats_file_path)
                
                if etiv is not None and etiv > 0:
                    # Calculate the percentage volume relative to eTIV for both source and target
                    source_volume_percent = (source_volume / etiv) * 100
                    target_volume_percent = (target_volume / etiv) * 100

                    # Save the percentage volume in the subject folder
                    subject_volume_dir = os.path.join(save_dir, subject_folder)
                    os.makedirs(subject_volume_dir, exist_ok=True)
                    
                    # Save source volume as percentage of eTIV
                    source_volume_file_path = os.path.join(subject_volume_dir, 'source_volume.txt')
                    with open(source_volume_file_path, 'w') as source_volume_file:
                        source_volume_file.write(f"Source Volume as percentage of eTIV: {source_volume_percent:.4f}%\n")
                    
                    # Save target volume as percentage of eTIV
                    target_volume_file_path = os.path.join(subject_volume_dir, 'target_volume.txt')
                    with open(target_volume_file_path, 'w') as target_volume_file:
                        target_volume_file.write(f"Target Volume as percentage of eTIV: {target_volume_percent:.4f}%\n")
                    
                    print(f"Saved source and target percentage volumes for {subject_folder}")
                else:
                    print(f"Invalid or missing eTIV value for {first_session_dir}")
            else:
                print(f"aseg.stats file not found for {first_session_dir} (even with \r)")
        else:
            print(f"Source or target segmentation file not found for {subject_folder}")
