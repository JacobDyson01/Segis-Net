import os
import nibabel as nib
import numpy as np
import sys

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

# Get inputs from the command line
input_dir = sys.argv[1]
output_dir = sys.argv[2]
result_filename = sys.argv[3]  # Example: "result.nii.gz"
cropped_filename = sys.argv[4]  # Example: "warped_roi.nii.gz"
crop_coords = tuple(map(int, sys.argv[5:]))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each subject directory
for sub_dir in os.listdir(input_dir):
    sub_id = os.path.basename(sub_dir)
    print(f"Processing {sub_id}...")

    # Define input and output paths
    src_img = os.path.join(input_dir, sub_dir, result_filename)
    src_roi = os.path.join(output_dir, sub_dir, cropped_filename)

    # Create output directory for the subject
    os.makedirs(os.path.join(output_dir, sub_id), exist_ok=True)

    # Apply cropping to the image
    crop_image(src_img, src_roi, crop_coords)

    print(f"{sub_id} processed and saved to {os.path.join(output_dir, sub_id)}")

print("Processing complete.")
