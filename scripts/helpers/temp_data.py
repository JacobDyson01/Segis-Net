import os
import shutil

# Directories
input_dirs = [
    '/home/groups/dlmrimnd/jacob/data/input_images',
    '/home/groups/dlmrimnd/jacob/data/input_affine_path',
    '/home/groups/dlmrimnd/jacob/data/deformation_fields'
]
output_base_dir = '/home/groups/dlmrimnd/jacob/data_temp'

# List of incompatible subject IDs
excluded_subjects = [
    "sub-041S6136", "sub-041S6354", "sub-024S6385", "sub-127S6348",
    "sub-016S6381", "sub-099S6016", "sub-041S6292", "sub-301S6224",
    "sub-941S6094", "sub-941S6054", "sub-041S6192", "sub-024S6005", "sub-024S6202"
]

# Function to copy directories excluding specific subjects
def copy_filtered_dirs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        # Extract the subject ID from the directory name
        subject_id = item.split('_')[0]
        # Check if the item is a directory and not in the excluded subjects
        if os.path.isdir(item_path) and subject_id not in excluded_subjects:
            shutil.copytree(item_path, os.path.join(output_dir, item))

# Copy contents for each input directory
for input_dir in input_dirs:
    output_dir = os.path.join(output_base_dir, os.path.basename(input_dir))
    copy_filtered_dirs(input_dir, output_dir)