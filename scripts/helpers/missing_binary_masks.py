import os
import nibabel as nib
import numpy as np

base_dir = '/home/groups/dlmrimnd/jacob/data/input_images'
source_dir = '/home/groups/dlmrimnd/akshit/ADNI_output_files'
output_dir = '/home/groups/dlmrimnd/jacob/data/binary_masks'

# Iterate over each subject
for sub_dir in os.listdir(base_dir):
    sub_path = os.path.join(base_dir, sub_dir)
    if not os.path.isdir(sub_path):
        continue

    # Extract session IDs from folder name
    subject_id, ses1, ses2 = sub_dir.split('_')
    
    # Correct session format for source directory
    ses1_source = ses1.replace('-', '_')
    ses2_source = ses2.replace('-', '_')

    # Define paths to the source and target directories in the source folder
    ses1_source_dir = os.path.join(source_dir, f'{subject_id}_{ses1_source}')
    ses2_source_dir = os.path.join(source_dir, f'{subject_id}_{ses2_source}')

    # Define the paths for binary masks in the output folder
    mask_ses1_dir = os.path.join(output_dir, f'{subject_id}_{ses1}')
    mask_ses2_dir = os.path.join(output_dir, f'{subject_id}_{ses2}')
    
    os.makedirs(mask_ses1_dir, exist_ok=True)
    os.makedirs(mask_ses2_dir, exist_ok=True)

    for ses_source_dir, mask_ses_dir in zip([ses1_source_dir, ses2_source_dir], [mask_ses1_dir, mask_ses2_dir]):
        if os.path.isdir(ses_source_dir):
            mgz_file_path = os.path.join(ses_source_dir, 'mri', 'aparc.DKTatlas+aseg.deep.withCC.mgz')
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
                nifti_file_path = os.path.join(mask_ses_dir, 'binary_mask.nii.gz')
                nib.save(binary_mask_img, nifti_file_path)

                print(f"Saved binary mask for {ses_source_dir} to {nifti_file_path}")