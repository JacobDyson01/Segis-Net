import os

# Directories to iterate through
warped_masks_dir = "/home/groups/dlmrimnd/jacob/data/warped_masks_roi"
output_dir = "/home/groups/dlmrimnd/akshit/ADNI_output_files"
save_dir = "/home/groups/dlmrimnd/jacob/data/volumes_ADNI"

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
                index = int(cols[0])
                if index in area_indices:
                    volume = float(cols[3])  # Assuming Volume_mm3 is the 4th column
                    total_volume += volume
            except (ValueError, IndexError):
                continue  # Skipping lines that are not valid
    return total_volume

# Iterate through each subject folder in the warped_masks_dir
for subject_folder in os.listdir(warped_masks_dir):
    
    subject_path = os.path.join(warped_masks_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        # First, try to find the corresponding folder without the carriage return
        adni_stats_folder = os.path.join(output_dir, subject_folder, 'stats')
        stats_file_path = os.path.join(adni_stats_folder, 'aseg+DKT.stats')
        
        # If the file doesn't exist, try appending '\r' to the folder name
        if not os.path.exists(stats_file_path):
            adni_stats_folder_with_r = os.path.join(output_dir, subject_folder + '\r', 'stats')
            stats_file_path = os.path.join(adni_stats_folder_with_r, 'aseg+DKT.stats')
        
        # Print the path being checked for the stats file
        print(f"Looking for stats file in: {stats_file_path}")
        
        if os.path.exists(stats_file_path):
            # Extract the total volume for the four brain areas
            total_volume = extract_volumes_from_stats(stats_file_path)
            
            # Create the output directory for volumes if it doesn't exist
            subject_volume_dir = os.path.join(save_dir, subject_folder)
            os.makedirs(subject_volume_dir, exist_ok=True)
            
            # Save the total volume to a file
            volume_file_path = os.path.join(subject_volume_dir, 'volume.txt')
            with open(volume_file_path, 'w') as volume_file:
                volume_file.write(f"Total volume: {total_volume} mm^3\n")
                
            print(f"Saved total volume for {subject_folder} to {volume_file_path}")
        else:
            print(f"Stats file not found for {subject_folder} or {subject_folder} '\r'")
