#!/bin/bash

#SBATCH --job-name=rigid_transform_missing
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/missing_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/missing_error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
base_dir="/home/groups/dlmrimnd/jacob/data/input_images"
rigid_warped_dir="/home/groups/dlmrimnd/jacob/data/rigid_input_to_sub"
rigid_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/rigid_transform_params.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"
template_img="/home/groups/dlmrimnd/jacob/data/input_images/sub-003S6067_ses-01_ses-02/source.nii.gz"  # Reference image

echo "Starting Elastix job for missing rigid transformations"

# List of specific subject directories to check
subject_dirs=(
  "sub-168S6049_ses-01_ses-02"
  "sub-168S6085_ses-01_ses-02"
  "sub-168S6151_ses-01_ses-02"
  "sub-168S6320_ses-01_ses-02"
  "sub-301S6224_ses-01_ses-02"
)

# Iterate over each specified subject directory
for subject_dir in "${subject_dirs[@]}"; do
  full_id="${subject_dir}"
  echo "Processing subject: ${full_id}"

  # Define paths to the images
  img1="${base_dir}/${full_id}/source.nii.gz"
  img2="${base_dir}/${full_id}/target.nii.gz"
  
  # Define output paths
  output_source="${rigid_warped_dir}/${full_id}/source_rigid.nii.gz"
  output_target="${rigid_warped_dir}/${full_id}/target_rigid.nii.gz"
  
  # Create output directory if it doesn't exist
  mkdir -p "${rigid_warped_dir}/${full_id}"

  # Check if the source image needs to be processed
  if [ ! -f "${output_source}" ]; then
    echo "Source image is missing for ${full_id}. Running transformation."
    singularity exec \
      -B ${img1}:${img1} \
      -B ${template_img}:${template_img} \
      -B ${rigid_warped_dir}:${rigid_warped_dir} \
      -B ${rigid_param_file}:${rigid_param_file} \
      ${elastix_sif} \
      elastix -threads 1 -f ${template_img} -m ${img1} -p ${rigid_param_file} -out "${rigid_warped_dir}/${full_id}"

    # Rename the output image for source
    mv "${rigid_warped_dir}/${full_id}/result.0.nii" "${output_source}"
  else
    echo "Source image already exists for ${full_id}. Skipping transformation."
  fi

  # Check if the target image needs to be processed
  if [ ! -f "${output_target}" ]; then
    echo "Target image is missing for ${full_id}. Running transformation."
    singularity exec \
      -B ${img2}:${img2} \
      -B ${template_img}:${template_img} \
      -B ${rigid_warped_dir}:${rigid_warped_dir} \
      -B ${rigid_param_file}:${rigid_param_file} \
      ${elastix_sif} \
      elastix -threads 1 -f ${template_img} -m ${img2} -p ${rigid_param_file} -out "${rigid_warped_dir}/${full_id}"

    # Rename the output image for target
    mv "${rigid_warped_dir}/${full_id}/result.0.nii" "${output_target}"
  else
    echo "Target image already exists for ${full_id}. Skipping transformation."
  fi

  echo "Processed ${full_id}"
done
