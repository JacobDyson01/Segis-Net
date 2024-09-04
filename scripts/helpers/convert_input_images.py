import os
import nibabel as nib

# Directory containing the subject folders with .nii.gz files
input_images_dir = '/home/groups/dlmrimnd/jacob/data/input_images_old'

# Source directory containing the .mgz files
source_dir = '/home/groups/dlmrimnd/akshit/ADNI_output_files'

# Target directory for the converted .nii.gz files
target_dir = '/home/groups/dlmrimnd/jacob/data/input_images'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Iterate over the subject directories in input_images
for subject_dir in os.listdir(input_images_dir):
    subject_id = subject_dir.split('_')[0]  # Extract the subject id (sub-xxx)
    ses1 = subject_dir.split('_')[1].replace('-', '_')  # Replace '-' with '_' for session id
    ses2 = subject_dir.split('_')[2].replace('-', '_')  # Replace '-' with '_' for session id
    
    # Define the source paths for the .mgz files
    src_path = os.path.join(source_dir, f'{subject_id}_{ses1}', 'mri', 'brain.mgz')
    tgt_path = os.path.join(source_dir, f'{subject_id}_{ses2}', 'mri', 'brain.mgz')
    
    if os.path.exists(src_path) and os.path.exists(tgt_path):
        target_sub_dir = os.path.join(target_dir, subject_dir)
        os.makedirs(target_sub_dir, exist_ok=True)
        
        # Convert source .mgz to .nii.gz
        src_img = nib.load(src_path)
        src_nii = nib.Nifti1Image(src_img.get_fdata(), src_img.affine)
        src_nii_path = os.path.join(target_sub_dir, 'source.nii.gz')
        nib.save(src_nii, src_nii_path)
        
        # Convert target .mgz to .nii.gz
        tgt_img = nib.load(tgt_path)
        tgt_nii = nib.Nifti1Image(tgt_img.get_fdata(), tgt_img.affine)
        tgt_nii_path = os.path.join(target_sub_dir, 'target.nii.gz')
        nib.save(tgt_nii, tgt_nii_path)
        
        print(f"Converted {src_path} to {src_nii_path}")
        print(f"Converted {tgt_path} to {tgt_nii_path}")
    else:
        print(f"Source or Target for {subject_id} does not exist. Skipping this subject.")
