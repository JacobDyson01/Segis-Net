#!/bin/bash

#SBATCH --job-name=deform
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
base_dir="/home/groups/dlmrimnd/jacob/data_storage/input_images_old"
affine_warped_dir="/home/groups/dlmrimnd/jacob/data/warped_full_head_to_sub"
affine_param_file="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/par_atlas_aff_checknrfalse.txt"
elastix_sif="/home/groups/dlmrimnd/jacob/files/Elastix_Preprocessing/elastix.sif"
template_img="/home/groups/dlmrimnd/jacob/data_storage/input_images_old/sub-003S6067_ses-01_ses-02/source.nii.gz"

echo "Starting Elastix job"

# Iterate over each subject
for sub_dir in ${base_dir}/sub-*; do
  full_id=$(basename ${sub_dir})
  echo "Processing subject: ${full_id}"

  # Extract subject ID and session ids from folder name
  sub_id=$(echo ${full_id} | grep -oP 'sub-\d+[A-Za-z\d]*')
  ses_ids=($(echo ${full_id} | grep -oP '(?<=ses-)\d+'))
  ses1_id="ses-${ses_ids[0]}"
  ses2_id="ses-${ses_ids[1]}"

  # Define paths to the images
  img1="${sub_dir}/source.nii.gz"
  img2="${sub_dir}/target.nii.gz"

  echo "Processing subject: ${sub_id} with sessions ${ses1_id} and ${ses2_id}"
  echo "Image 1 path: ${img1}"
  echo "Image 2 path: ${img2}"

  # Define output directory for warped images
  warped_output_dir="${affine_warped_dir}/${full_id}"
  mkdir -p ${warped_output_dir}

  # Run Elastix using Singularity to generate the deformation field from source to template
  singularity exec \
    -B ${img1}:${img1} \
    -B ${template_img}:${template_img} \
    -B ${warped_output_dir}:${warped_output_dir} \
    -B ${affine_param_file}:${affine_param_file} \
    ${elastix_sif} \
    elastix -threads 1 -f ${template_img} -m ${img1} -p ${affine_param_file} -out ${warped_output_dir}

  # Rename the TransformParameters file for source
  mv ${warped_output_dir}/TransformParameters.0.txt ${warped_output_dir}/TransformParameters_${sub_id}_${ses1_id}.txt

  # Generate the warped source image using Transformix
  singularity exec \
    -B ${img1}:${img1} \
    -B ${warped_output_dir}:${warped_output_dir} \
    ${elastix_sif} \
    transformix -in ${img1} -tp ${warped_output_dir}/TransformParameters_${sub_id}_${ses1_id}.txt -out ${warped_output_dir}

  # Rename the warped source image
  mv ${warped_output_dir}/result.nii.gz ${warped_output_dir}/warped_source_${ses1_id}.nii.gz

  # Run Elastix using Singularity to generate the deformation field from target to template
  singularity exec \
    -B ${img2}:${img2} \
    -B ${template_img}:${template_img} \
    -B ${warped_output_dir}:${warped_output_dir} \
    -B ${affine_param_file}:${affine_param_file} \
    ${elastix_sif} \
    elastix -threads 1 -f ${template_img} -m ${img2} -p ${affine_param_file} -out ${warped_output_dir}

  # Rename the TransformParameters file for target
  mv ${warped_output_dir}/TransformParameters.0.txt ${warped_output_dir}/TransformParameters_${sub_id}_${ses2_id}.txt

  # Generate the warped target image using Transformix
  singularity exec \
    -B ${img2}:${img2} \
    -B ${warped_output_dir}:${warped_output_dir} \
    ${elastix_sif} \
    transformix -in ${img2} -tp ${warped_output_dir}/TransformParameters_${sub_id}_${ses2_id}.txt -out ${warped_output_dir}

  # Rename the warped target image
  mv ${warped_output_dir}/result.nii.gz ${warped_output_dir}/warped_target_${ses2_id}.nii.gz

  echo "Processed ${full_id}"
done
