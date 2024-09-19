import os
import nibabel as nib
import numpy as np

# Specify the patient's session directory
session_dir = '/home/groups/dlmrimnd/akshit/output_files/sub-003S6014_ses_01'

# Desired output directory for NIfTI files
output_dir = '/home/groups/dlmrimnd/jacob/data/nifti_files/sub-003S6014_ses_01'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the path to the mri subfolder
mri_dir = os.path.join(session_dir, 'mri')
mgz_file_path = os.path.join(mri_dir, 'aparc.DKTatlas+aseg.deep.withCC.mgz')

if os.path.exists(mgz_file_path):
    # Load the MGZ file
    img = nib.load(mgz_file_path)
    
    # Get the image data as a NumPy array
    img_data = img.get_fdata()
    
    # Apply the threshold to isolate specific brain regions
    precentral_gyrus_mask = (img_data == 1024) | (img_data == 2024)
    paracentral_lobule_mask = (img_data == 2017) | (img_data == 1017)
    
    # Combine masks for the specific brain regions
    combined_mask = precentral_gyrus_mask | paracentral_lobule_mask
    
    # Convert to binary mask (1 for the regions of interest, 0 for everything else)
    binary_mask = combined_mask.astype(np.uint8)
    
    # Save the binary mask as a NIfTI file
    binary_mask_img = nib.Nifti1Image(binary_mask, img.affine)
    nifti_file_path = os.path.join(output_dir, 'binary_mask.nii.gz')
    nib.save(binary_mask_img, nifti_file_path)
    
    print(f'Processed {mgz_file_path} and saved binary mask at {nifti_file_path}')
else:
    print(f'MGZ file not found at {mgz_file_path}')