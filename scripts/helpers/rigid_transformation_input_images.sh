#!/bin/bash

#SBATCH --job-name=rigid_transform
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
base_dir="/home/groups/dlmrimnd/jacob/data_storage/input_images_old"
rigid_warped_dir="/home/groups/dlmrimnd/jacob/data/rigid_whole_head_to_sub"
rigid_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/rigid_transform_params.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"
template_img="/home/groups/dlmrimnd/jacob/data/input_images/sub-003S6067_ses-01_ses-02/source.nii.gz"  # Reference image

echo "Starting Elastix job for rigid transformations"

# Iterate over each subject
for sub_dir in ${base_dir}/sub-*; do
  full_id=$(basename ${sub_dir})
  echo "Processing subject: ${full_id}"

  # Define paths to the images
  img1="${sub_dir}/source.nii.gz"
  img2="${sub_dir}/target.nii.gz"

  echo "Image 1 path: ${img1}"
  echo "Image 2 path: ${img2}"

  # Define output directory for warped images
  warped_output_dir="${rigid_warped_dir}/${full_id}"
  mkdir -p ${warped_output_dir}

  # Run Elastix using Singularity to perform rigid transformation from source to template
  singularity exec \
    -B ${img1}:${img1} \
    -B ${template_img}:${template_img} \
    -B ${warped_output_dir}:${warped_output_dir} \
    -B ${rigid_param_file}:${rigid_param_file} \
    ${elastix_sif} \
    elastix -threads 1 -f ${template_img} -m ${img1} -p ${rigid_param_file} -out ${warped_output_dir}

  # Rename the output image for source
  mv ${warped_output_dir}/result.0.nii ${warped_output_dir}/source_rigid.nii.gz

  # Run Elastix using Singularity to perform rigid transformation from target to template
  singularity exec \
    -B ${img2}:${img2} \
    -B ${template_img}:${template_img} \
    -B ${warped_output_dir}:${warped_output_dir} \
    -B ${rigid_param_file}:${rigid_param_file} \
    ${elastix_sif} \
    elastix -threads 1 -f ${template_img} -m ${img2} -p ${rigid_param_file} -out ${warped_output_dir}

  # Rename the output image for target
  mv ${warped_output_dir}/result.0.nii ${warped_output_dir}/target_rigid.nii.gz

  echo "Processed ${full_id}"
done
