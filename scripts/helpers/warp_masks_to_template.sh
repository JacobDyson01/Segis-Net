#!/bin/bash

#SBATCH --job-name=warp_masks
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
mask_dir="/home/groups/dlmrimnd/jacob/data/binary_masks"
mask_output_dir="/home/groups/dlmrimnd/jacob/data/warped_masks"
affine_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/par_atlas_aff_checknrfalse_binary.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"
template_img="/home/groups/dlmrimnd/jacob/data/brain_mni_256.nii.gz"  # Path to your template image

echo "Starting Elastix job"

# Iterate over each subject
for sub_dir in ${mask_dir}/sub-*; do
  sub_id=$(basename ${sub_dir})
  echo "Processing subject: ${sub_id}"

  # Define the path to the binary mask
  mask_img="${mask_dir}/${sub_id}/binary_mask.nii.gz"
  mask_output="${mask_output_dir}/${sub_id}/warped_mask.nii.gz"

  # Create the output directory for the warped mask
  mkdir -p $(dirname ${mask_output})

  # Run Elastix using Singularity to generate the deformation field from mask to template
  singularity exec \
    -B ${mask_img}:${mask_img} \
    -B ${template_img}:${template_img} \
    -B $(dirname ${mask_output}):$(dirname ${mask_output}) \
    -B ${affine_param_file}:${affine_param_file} \
    ${elastix_sif} \
    elastix -threads 1 -f ${template_img} -m ${mask_img} -p ${affine_param_file} -out $(dirname ${mask_output})

  # Generate the warped mask image using Transformix
  singularity exec \
    -B ${mask_img}:${mask_img} \
    -B $(dirname ${mask_output}):$(dirname ${mask_output}) \
    ${elastix_sif} \
    transformix -in ${mask_img} -tp $(dirname ${mask_output})/TransformParameters.0.txt -out $(dirname ${mask_output})

  # Rename the warped mask image
  mv $(dirname ${mask_output})/result.nii.gz ${mask_output}

  echo "Processed ${sub_id}"
done
