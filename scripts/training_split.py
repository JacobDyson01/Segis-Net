import os
import numpy as np
from collections import defaultdict

# Set the base directory for the data
data_dir = '/home/groups/dlmrimnd/jacob/data/combined_data'

# Path to the input images directory
input_images_dir = os.path.join(data_dir, 'warped_input_roi')

# Dictionary to group subjects by patient ID
patient_sessions = defaultdict(list)

# List of all subject folders
subject_folders = os.listdir(input_images_dir)

# Group sessions by patient ID
for folder in subject_folders:
    patient_id = folder.split('_')[0]  # Extract the patient ID (e.g., sub-003S6067)
    patient_sessions[patient_id].append(folder)

# Shuffle the patient IDs
patient_ids = list(patient_sessions.keys())
np.random.seed(42)
np.random.shuffle(patient_ids)

# Define split ratio
split_ratio = 0.2
split_index = int(len(patient_ids) * split_ratio)

# Split the patient IDs into training and validation sets
vali_patient_ids = patient_ids[:split_index]
train_patient_ids = patient_ids[split_index:]

# Get corresponding subject folders for each split
train_subjects = [session for patient_id in train_patient_ids for session in patient_sessions[patient_id]]
vali_subjects = [session for patient_id in vali_patient_ids for session in patient_sessions[patient_id]]

# Print the training subjects and their sessions
print("Training set subjects and sessions:")
for patient_id in train_patient_ids:
    print(f"Patient {patient_id}: Sessions: {patient_sessions[patient_id]}")

# Print the validation subjects and their sessions
print("\nValidation set subjects and sessions:")
for patient_id in vali_patient_ids:
    print(f"Patient {patient_id}: Sessions: {patient_sessions[patient_id]}")

# Save the indices to .npy files
np.save(os.path.join(data_dir, 'train_index_new.npy'), train_subjects)
np.save(os.path.join(data_dir, 'vali_index_new.npy'), vali_subjects)

# Confirm the split
print(f"\nTotal number of training subjects: {len(train_patient_ids)}")
print(f"Total number of validation subjects: {len(vali_patient_ids)}")
print(f"Training set sessions: {len(train_subjects)}")
print(f"Validation set sessions: {len(vali_subjects)}")
