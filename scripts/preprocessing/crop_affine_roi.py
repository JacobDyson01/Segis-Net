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
crop_coords = (3, 195, 20, 212, 6, 182)

# Directories
input_dir = "/home/groups/dlmrimnd/jacob/data/upgraded_segis_data/warped_input"
output_dir = "/home/groups/dlmrimnd/jacob/data/upgraded_segis_data/warped_input_roi"  

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each subject directory
for sub_dir in os.listdir(input_dir):
    sub_dir_path = os.path.join(input_dir, sub_dir)
    
    if not os.path.isdir(sub_dir_path):
        # Skip any non-directory items (if they exist)
        continue

    print(f"Processing {sub_dir}...")

    # Define input and output paths
    src_img = os.path.join(sub_dir_path, "source.nii.gz")
    tgt_img = os.path.join(sub_dir_path, "target.nii.gz")

    # Define output paths and ensure the output directory exists for the subject
    output_sub_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(output_sub_dir, exist_ok=True)  # Ensure the output folder for the subject is created

    src_roi = os.path.join(output_sub_dir, "source_roi.nii.gz")
    tgt_roi = os.path.join(output_sub_dir, "target_roi.nii.gz")

    # Apply cropping to source and target images
    if os.path.exists(src_img):
        crop_image(src_img, src_roi, crop_coords)
    else:
        print(f"Source image {src_img} does not exist.")
    crop_image(tgt_img, tgt_roi, crop_coords)
    print(f"{sub_dir} processed and saved to {output_sub_dir}")

print("Processing complete.")
