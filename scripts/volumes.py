import os

# Directories to iterate through
warped_masks_dir = "/home/groups/dlmrimnd/jacob/data/MND/transformation_matrices"
output_dir = "/home/groups/dlmrimnd/akshit/MND_output_files"
save_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_MND"

# Brain area indices
area_indices = {
    60: 'lh-precentral',
    53: 'lh-paracentral',
    91: 'rh-precentral',
    84: 'rh-paracentral'
}

# Function to extract volume from aseg+DKT.stats file for the specific brain areas
def extract_volumes_from_stats(file_path):
    total_volume = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            cols = line.split()
            try:
                # Extract brain area volumes
                index = int(cols[0])
                if index in area_indices:
                    volume = float(cols[3])  # Assuming Volume_mm3 is the 4th column
                    total_volume += volume
            except (ValueError, IndexError):
                continue  # Skipping lines that are not valid
    return total_volume

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

# Iterate through each subject folder in the warped_masks_dir
for subject_folder in os.listdir(warped_masks_dir):
    
    subject_path = os.path.join(warped_masks_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        # First, try to find the corresponding folder without the carriage return
        adni_stats_folder = os.path.join(output_dir, subject_folder, 'stats')
        stats_file_path = os.path.join(adni_stats_folder, 'aseg+DKT.stats')
        aseg_stats_file_path = os.path.join(adni_stats_folder, 'aseg.stats')  # Path to aseg.stats
        
        # If the file doesn't exist, try appending '\r' to the folder name
        if not os.path.exists(stats_file_path):
            adni_stats_folder_with_r = os.path.join(output_dir, subject_folder + '\r', 'stats')
            stats_file_path = os.path.join(adni_stats_folder_with_r, 'aseg+DKT.stats')
            aseg_stats_file_path = os.path.join(adni_stats_folder_with_r, 'aseg.stats')  # Path to aseg.stats

        # Print the paths being checked for the stats files
        print(f"Looking for stats file in: {stats_file_path} and {aseg_stats_file_path}")
        
        if os.path.exists(stats_file_path) and os.path.exists(aseg_stats_file_path):
            # Extract the total volume from the aseg+DKT.stats file
            total_volume = extract_volumes_from_stats(stats_file_path)
            
            # Extract the eTIV from the aseg.stats file
            etiv = extract_etiv_from_aseg(aseg_stats_file_path)
            
            if etiv is not None and etiv > 0:
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
                print(f"eTIV value not found or invalid for {subject_folder}")
        else:
            print(f"Stats file not found for {subject_folder}")
