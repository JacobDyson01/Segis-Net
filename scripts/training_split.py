import os
import numpy as np

# Set the base directory for the data
data_dir = '/home/groups/dlmrimnd/jacob/data_temp'

# Path to the input images directory
input_images_dir = os.path.join(data_dir, 'input_images')

# List of all subjects
subjects = os.listdir(input_images_dir)

# Shuffle the subjects
np.random.seed(42)
np.random.shuffle(subjects)

# Define split ratio
split_ratio = 0.2

# Split the subjects into training and validation sets
split_index = int(len(subjects) * split_ratio)
vali_subjects = subjects[:split_index]
train_subjects = subjects[split_index:]

# Save the indices to .npy files
np.save(os.path.join(data_dir, 'train_index.npy'), train_subjects)
np.save(os.path.join(data_dir, 'vali_index.npy'), vali_subjects)