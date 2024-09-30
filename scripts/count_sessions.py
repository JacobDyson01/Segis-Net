import os

# Define the directory that contains the folders
directory = '/home/groups/dlmrimnd/jacob/data/warped_masks'

# Dictionary to keep track of patient session counts
patient_counts = {}

# Iterate through the directory structure
for folder_name in os.listdir(directory):
    if folder_name.startswith('sub') and 'ses' in folder_name:
        # Extract the patient ID (up to the first underscore)
        patient_id = folder_name.split('_')[0]
        # Increment the count of sessions for the patient
        if patient_id in patient_counts:
            patient_counts[patient_id] += 1
        else:
            patient_counts[patient_id] = 1

# Dictionary to store the final counts of patients with specific number of scans
scan_counts = {}

# Populate the scan counts dictionary
for patient, count in patient_counts.items():
    if count not in scan_counts:
        scan_counts[count] = 1
    else:
        scan_counts[count] += 1

# Output the results
for scans, num_patients in sorted(scan_counts.items()):
    print(f'Patients with {scans} scans: {num_patients}')
