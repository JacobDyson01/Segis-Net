import nibabel as nib
import numpy as np

# Define the file paths
mask1_path = "/home/groups/dlmrimnd/jacob/data/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
mask2_path = "/home/groups/dlmrimnd/jacob/data/mni_icbm152_t1_tal_nlin_sym_09a.nii"
output_path = "/home/groups/dlmrimnd/jacob/data/brain_mni.nii.gz"

# Load the NIfTI images
mask1_img = nib.load(mask1_path)
mask2_img = nib.load(mask2_path)

# Extract the data arrays from the NIfTI images
mask1_data = mask1_img.get_fdata()
mask2_data = mask2_img.get_fdata()

# Multiply the data arrays together
multiplied_data = mask1_data * mask2_data

# Create a new NIfTI image with the multiplied data and the original affine matrix
multiplied_img = nib.Nifti1Image(multiplied_data, mask1_img.affine)

# Save the resulting NIfTI image
nib.save(multiplied_img, output_path)

print(f"Multiplied mask saved to {output_path}")