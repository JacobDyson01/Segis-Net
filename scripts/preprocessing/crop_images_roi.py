import os
import nibabel as nib
import numpy as np

def crop_image(input_path, output_path, crop_coords):
    # Load the image
    img = nib.load(input_path)
    img_data = img.get_fdata()

    # Crop the image using the specified coordinates
    x_min, x_max, y_min, y_max, z_min, z_max = crop_coords
    cropped_img_data = img_data[x_min:x_max, y_min:y_max, z_min:z_max]

    # Save the cropped image
    cropped_img = nib.Nifti1Image(cropped_img_data, img.affine)
    nib.save(cropped_img, output_path)

# Define the cropping coordinates (adjust these values as needed)
crop_coords = (12, 188, 80, 160, 60, 156)

# Directories
input_dir = "/home/groups/dlmrimnd/jacob/data/MND/warped_input"
output_dir = "/home/groups/dlmrimnd/jacob/data/MND/warped_input_roi"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each subject directory
for sub_dir in os.listdir(input_dir):
    sub_id = os.path.basename(sub_dir)
    print(f"Processing {sub_id}...")

    # Define input and output paths
    src_img = os.path.join(input_dir, sub_dir, "warped_source_Warped.nii.gz")
    tgt_img = os.path.join(input_dir, sub_dir, "warped_target_Warped.nii.gz")
    src_roi = os.path.join(output_dir, sub_dir, "source_Warped_roi.nii.gz")
    tgt_roi = os.path.join(output_dir, sub_dir, "target_Warped_roi.nii.gz")

    # Create output directory for subject
    os.makedirs(os.path.join(output_dir, sub_id), exist_ok=True)

    # Apply cropping to source and target images
    crop_image(src_img, src_roi, crop_coords)
    crop_image(tgt_img, tgt_roi, crop_coords)

    print(f"{sub_id} processed and saved to {os.path.join(output_dir, sub_id)}")

print("Processing complete.")