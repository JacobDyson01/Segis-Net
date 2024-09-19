import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

input_dir = "/home/groups/dlmrimnd/jacob/data/input_images"
output_dir = "/home/groups/dlmrimnd/jacob/data/input_images_downsampled"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def downsample_image(input_path, output_path, scale_factor=0.625):
    # Load the image
    img = nib.load(input_path)
    img_data = img.get_fdata()

    # Downsample the image
    zoom_factors = [scale_factor] * len(img_data.shape)
    downsampled_data = zoom(img_data, zoom_factors, order=1)

    # Adjust the voxel size according to the scale factor
    new_affine = np.copy(img.affine)
    new_affine[:3, :3] *= 1 / scale_factor

    # Save the downsampled image with the new voxel size
    downsampled_img = nib.Nifti1Image(downsampled_data, new_affine)
    nib.save(downsampled_img, output_path)

# Iterate over each subject directory
for sub_dir in os.listdir(input_dir):
    sub_path = os.path.join(input_dir, sub_dir)
    if os.path.isdir(sub_path):
        print(f"Processing {sub_dir}...")

        # Define paths to the input and output images
        src_img = os.path.join(sub_path, "source.nii.gz")
        tgt_img = os.path.join(sub_path, "target.nii.gz")
        src_out_dir = os.path.join(output_dir, sub_dir)
        tgt_out_dir = src_out_dir

        # Create the output directory
        os.makedirs(src_out_dir, exist_ok=True)

        # Downsample the images
        downsample_image(src_img, os.path.join(src_out_dir, "source.nii.gz"))
        downsample_image(tgt_img, os.path.join(tgt_out_dir, "target.nii.gz"))

        print(f"{sub_dir} processed and saved.")

print("All subjects processed.")