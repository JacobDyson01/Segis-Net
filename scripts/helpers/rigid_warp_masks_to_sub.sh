#!/bin/bash

#SBATCH --job-name=rigid_transform_masks
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/mask_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/mask_error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
mask_dir="/home/groups/dlmrimnd/jacob/data/binary_masks"
rigid_mask_dir="/home/groups/dlmrimnd/jacob/data/rigid_masks"
rigid_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/rigid_transform_params_mask.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"
template_img="/home/groups/dlmrimnd/jacob/data/input_images/sub-003S6067_ses-01_ses-02/source.nii.gz"  # Reference image

echo "Starting Elastix job for rigid transformation of binary masks"

# Iterate over each subject
for sub_dir in ${mask_dir}/sub-*; do
  full_id=$(basename ${sub_dir})
  echo "Processing subject: ${full_id}"

  # Define paths to the binary mask
  mask_img="${sub_dir}/binary_mask.nii.gz"

  # Define output directory for warped masks
  warped_output_dir="${rigid_mask_dir}/${full_id}"
  mkdir -p ${warped_output_dir}

  # Run Elastix using Singularity to perform rigid transformation from mask to template
  singularity exec \
    -B ${mask_img}:${mask_img} \
    -B ${template_img}:${template_img} \
    -B ${warped_output_dir}:${warped_output_dir} \
    -B ${rigid_param_file}:${rigid_param_file} \
    ${elastix_sif} \
    elastix -threads 1 -f ${template_img} -m ${mask_img} -p ${rigid_param_file} -out ${warped_output_dir}

  # Rename the warped mask image
  mv ${warped_output_dir}/result.0.nii ${warped_output_dir}/warped_binary_mask.nii.gz

  echo "Processed ${full_id}"
done
