import nibabel as nib

# Load MRI scan and segmentation
mri_scan = nib.load('/home/groups/dlmrimnd/jacob/data/MND/warped_input_roi/sub-005MND_ses-04_ses-05/source_Warped_roi.nii.gz')
segmentation = nib.load('/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_final/sub-005MND_ses-04_ses-06/source_segmentation.nii.gz')

# Check the shapes and affine matrices
print(f"MRI shape: {mri_scan.shape}, Segmentation shape: {segmentation.shape}")
print(f"MRI affine:\n{mri_scan.affine}\nSegmentation affine:\n{segmentation.affine}")

# If shapes don't match, resample the segmentation to match the MRI
# You can use packages like nilearn for this, if necessary
