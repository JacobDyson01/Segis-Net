import os
import nibabel as nib
import numpy as np
import sys

# Get input directories from command line arguments
root_dir = sys.argv[1]
output_dir = sys.argv[2]

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Walk through the root directory to find the mri subfolder
for subdir in os.listdir(root_dir):
    session_dir = os.path.join(root_dir, subdir)
    
    # Check if the session_dir is indeed a directory
    if os.path.isdir(session_dir):
        # Count the number of sessions
        session_count = len([s for s in os.listdir(root_dir) if s.startswith(subdir.split('_')[0])])
        
        # Process only if the patient has at least two sessions
        if session_count >= 2:
            mri_dir = os.path.join(session_dir, 'mri')
            
            if os.path.isdir(mri_dir):
                mgz_file_path = os.path.join(mri_dir, 'aparc.DKTatlas+aseg.deep.withCC.mgz')
                
                if os.path.exists(mgz_file_path):
                    # Create the output directory for the current subject session
                    nifti_subdir = os.path.join(output_dir, subdir)
                    os.makedirs(nifti_subdir, exist_ok=True)
                    
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
                    nifti_file_path = os.path.join(nifti_subdir, 'binary_mask.nii.gz')
                    nib.save(binary_mask_img, nifti_file_path)
