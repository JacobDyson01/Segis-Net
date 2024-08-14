#!/bin/bash

#SBATCH --job-name=elastix_job
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_%j.err
#SBATCH --partition=cpu

# Set the base directories
base_dir="/home/groups/dlmrimnd/jacob/data_temp/input_images"
output_base_dir="/home/groups/dlmrimnd/jacob/data_temp/deformation_fields"
affine_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/par_atlas_aff_checknrfalse.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"

# Define subject ID
sub_id="sub-067S6443_ses-01_ses-02"
sub_dir="${base_dir}/${sub_id}"

echo "Processing subject: ${sub_id}"

# Define paths to the images
img1="${sub_dir}/source.nii.gz"
img2="${sub_dir}/target.nii.gz"

echo "Image 1 path: ${img1}"
echo "Image 2 path: ${img2}"

# Define output directory
deform_dir="${output_base_dir}/${sub_id}"
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