#!/bin/bash

#SBATCH --job-name=warp_masks
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
mask_dir="/home/groups/dlmrimnd/jacob/data/binary_masks"
mask_output_dir="/home/groups/dlmrimnd/jacob/data/warped_masks_sub"
transform_param_dir="/home/groups/dlmrimnd/jacob/data/transformParameters"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"


echo "Starting Transformix job"

# Iterate over each subject
for sub_dir in ${mask_dir}/sub-*; do
  sub_id=$(basename ${sub_dir})
  echo "Processing subject: ${sub_id}"

  # Extract session ID from the subject ID
  ses_id=$(echo ${sub_id} | grep -oP 'ses-\d+')

  # Define the path to the corresponding TransformParameters file
  transform_param_file="${transform_param_dir}/TransformParameters_${sub_id}.txt"

  # Define the path to the binary mask
  mask_img="${mask_dir}/${sub_id}/binary_mask.nii.gz"
  mask_output="${mask_output_dir}/${sub_id}/warped_mask.nii.gz"

  # Create the output directory for the warped mask
  mkdir -p $(dirname ${mask_output})

  # Generate the warped mask image using Transformix
  singularity exec \
    -B ${mask_img}:${mask_img} \
    -B $(dirname ${mask_output}):$(dirname ${mask_output}) \
    ${elastix_sif} \
    transformix -in ${mask_img} -tp ${transform_param_file} -out $(dirname ${mask_output})

  # Rename the warped mask image
  mv $(dirname ${mask_output})/result.nii.gz ${mask_output}

  echo "Processed ${sub_id}"
done
