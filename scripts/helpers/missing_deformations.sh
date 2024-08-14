#!/bin/bash

#SBATCH --job-name=elastix_job_continue
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_continue_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_continue_%j.err
#SBATCH --partition=cpu

# Set the base directories
base_dir="/home/groups/dlmrimnd/jacob/data/input_images"
output_base_dir="/home/groups/dlmrimnd/jacob/data/deformation_fields"
affine_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/par_atlas_aff_checknrfalse.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"

echo "Starting Elastix job"

# Iterate over each subject
for sub_dir in ${base_dir}/sub-*; do
  sub_id=$(basename ${sub_dir})
  deform_dir="${output_base_dir}/${sub_id}"

  # Check if the deformation field directory already exists
  if [ ! -d "${deform_dir}" ]; then
    echo "Processing subject: ${sub_id}"

    # Extract session ids from folder name
    ses_ids=($(echo ${sub_id} | grep -oP '(?<=ses)\d+'))

    ses1_id="ses${ses_ids[0]}"
    ses2_id="ses${ses_ids[1]}"

    # Define paths to the images
    img1="${sub_dir}/source.nii.gz"
    img2="${sub_dir}/target.nii.gz"

    echo "Processing sessions: ${ses1_id} and ${ses2_id}"
    echo "Image 1 path: ${img1}"
    echo "Image 2 path: ${img2}"

    # Create output directory
    mkdir -p ${deform_dir}

    # Run Elastix using Singularity
    singularity exec \
      -B ${img1}:${img1} \
      -B ${img2}:${img2} \
      -B ${deform_dir}:${deform_dir} \
      -B ${affine_param_file}:${affine_param_file} \
      ${elastix_sif} \
      elastix -threads 1 -f ${img1} -m ${img2} -p ${affine_param_file} -out ${deform_dir}

    # Generate the deformation field using Transformix
    singularity exec \
      -B ${img2}:${img2} \
      -B ${deform_dir}:${deform_dir} \
      ${elastix_sif} \
      transformix -def all -tp ${deform_dir}/TransformParameters.0.txt -out ${deform_dir}

    echo "Processed ${sub_id}"
  else
    echo "Skipping ${sub_id}, deformation field already exists."
  fi
done
