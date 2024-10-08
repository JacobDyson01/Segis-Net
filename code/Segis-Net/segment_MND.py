import os
import nibabel as nib
import numpy as np
from keras.models import load_model
from SegisNet_model_dataGenerator import joint_model

# File paths
model_weights_path = '/home/groups/dlmrimnd/jacob/data/combined_data/saved_results/run_proper_1/model_weight_out.h5'
input_data_dir = '/home/groups/dlmrimnd/jacob/data/MND/warped_input_roi'
affine_data_dir = '/home/groups/dlmrimnd/jacob/data/MND/deformation_fields_roi_real'  # Corrected path for affine deformations
output_segmentation_dir = '/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_new'

# Create output directory if it doesn't exist
if not os.path.exists(output_segmentation_dir):
    os.makedirs(output_segmentation_dir)

# Parameters for the input data
params_inference = {
    'dim_xyz': (176, 80, 96),  # Example dimensions, adjust to match your data
    'R_ch': 1,
    'S_ch': 1,
    'batch_size': 1,
    'shuffle': False  # No need to shuffle during inference
}

# Initialize the Segis-Net model
print(f"Loading model weights from: {model_weights_path}")
model = joint_model(params_inference['dim_xyz'], params_inference['R_ch'], params_inference['S_ch'], 1, indexing='ij', alpha=0.2)
# model.load_weights(model_weights_path, by_name=True)
model.load_weights(model_weights_path)
print("Model weights loaded successfully.")

# Loop through the subfolders in the input directory
for subject_dir in os.listdir(input_data_dir):
    subject_path = os.path.join(input_data_dir, subject_dir)
    
    # Check if this is a directory containing the necessary NIfTI files
    if os.path.isdir(subject_path):
        print(f"Processing subject: {subject_dir}")
        
        source_file = os.path.join(subject_path, 'source_Warped_roi.nii.gz')
        target_file = os.path.join(subject_path, 'target_Warped_roi.nii.gz')
        
        # For affine directories, replace '-' with '_' to match the format in the affine directories
        # affine_subject_dir = subject_dir.replace('ses-', 'ses_')
        affine_subject_dir = subject_dir
        affine_file = os.path.join(affine_data_dir, affine_subject_dir, 'deformation_1Warp_roi.nii.gz')

        # Check if all files exist before proceeding
        if not os.path.exists(source_file):
            print(f"Source file not found for {subject_dir}: {source_file}")
            continue  # Skip to the next subject if file is missing
        if not os.path.exists(target_file):
            print(f"Target file not found for {subject_dir}: {target_file}")
            continue  # Skip to the next subject if file is missing
        if not os.path.exists(affine_file):
            print(f"Affine file not found for {subject_dir} (looked for {affine_subject_dir}): {affine_file}")
            continue  # Skip to the next subject if file is missing
        
        # Load the source, target, and affine images
        print(f"Loading source, target, and affine images for {subject_dir}")
        source_img = nib.load(source_file).get_fdata()
        target_img = nib.load(target_file).get_fdata()
        affine_img = nib.load(affine_file).get_fdata()
        
        # Assuming the segmentation source input is the same as the source image
        S_src = source_img
        
        # Reshape the data to match the model's input shape
        source_data = np.expand_dims(source_img, axis=[0, -1])  # Add batch and channel dimensions
        target_data = np.expand_dims(target_img, axis=[0, -1])  # Add batch and channel dimensions
        S_src_data = np.expand_dims(S_src, axis=[0, -1])        # Add batch and channel dimensions
        # affine_data = np.expand_dims(affine_img, axis=[0, -1])  # Add batch and channel dimensions
        affine_data = np.squeeze(affine_img)
        affine_data = np.expand_dims(affine_data, axis=[0])
        # Prepare the inputs as a list, since the model expects multiple inputs
        inputs = [target_data, source_data, S_src_data, affine_data]

        # Predict segmentation output
        print(f"Running segmentation for {subject_dir}")
        predicted_segmentation = model.predict(inputs)
        
        # predicted_segmentation[1] = tgt_segm (transformed source segmentation)
        # predicted_segmentation[2] = src_segm (segmentation of the source image)
        
        # Save the target segmentation (tgt_segm) and source segmentation (src_segm)
        subject_output_dir = os.path.join(output_segmentation_dir, subject_dir)
        if not os.path.exists(subject_output_dir):
            os.makedirs(subject_output_dir)
        
        # Save target segmentation
        target_segmentation_file = os.path.join(subject_output_dir, 'target_segmentation.nii.gz')
        nib.save(nib.Nifti1Image(predicted_segmentation[1][0, :, :, :, 0], np.eye(4)), target_segmentation_file)
        print(f"Saved target segmentation for {subject_dir} at {target_segmentation_file}")
        
        # Save source segmentation
        source_segmentation_file = os.path.join(subject_output_dir, 'source_segmentation.nii.gz')
        nib.save(nib.Nifti1Image(predicted_segmentation[2][0, :, :, :, 0], np.eye(4)), source_segmentation_file)
        print(f"Saved source segmentation for {subject_dir} at {source_segmentation_file}")
        
    else:
        print(f"Skipped {subject_dir}, not a directory")
