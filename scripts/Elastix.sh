#!/bin/bash

# Load necessary modules
# module unload fsl
# module load fsl/4.1.4

# Get the subject name from the argument
subjname=$1

# Set the base directory
base_dir="/home/groups/dlmrimnd/akshit/data/ADNI_HIPPO_T1w_ONLY"

# Set the base directory for output data
output_base_dir="/home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing"

# Define the paths for rigid and affine parameter files
affine_param_file="/home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing/par_atlas_aff_checknrfalse.txt"

# Iterate over each subject
for sub_dir in ${base_dir}/sub-*; do
  sub_id=$(basename ${sub_dir})
  echo "Processing subject: ${sub_id}"

  # Get a list of sessions for the subject
  sessions=(${sub_dir}/ses-*)

  # Ensure there are at least two sessions
  if [ ${#sessions[@]} -lt 2 ]; then
    echo "Not enough sessions for ${sub_id}. Skipping."
    continue
  fi

  # Process each combination of pairs of sessions
  for ((i=0; i<${#sessions[@]}-1; i++)); do
    for ((j=i+1; j<${#sessions[@]}; j++)); do
      session1=${sessions[$i]}
      session2=${sessions[$j]}

      ses1_id=$(basename ${session1})
      ses2_id=$(basename ${session2})

      # Define paths to the images
      img1="${session1}/anat/${sub_id}_${ses1_id}_T1w.nii.gz"
      img2="${session2}/anat/${sub_id}_${ses2_id}_T1w.nii.gz"

      # Define output directories
      deform_dir="${output_base_dir}/Preprocess_affine_transformParam/${sub_id}_${ses1_id}_${ses2_id}"
      dest_def_dir="${output_base_dir}/Preprocess_affine_deform_map/${sub_id}_${ses1_id}_${ses2_id}"
      dest_img_dir="${output_base_dir}/Preprocess_affine_input_img/${sub_id}_${ses1_id}_${ses2_id}"

      mkdir -p ${deform_dir}
      mkdir -p ${dest_def_dir}
      mkdir -p ${dest_img_dir}

      # Copy the images to the destination directory
      cp ${img1} ${dest_img_dir}/tgt_FA.nii.gz
      cp ${img2} ${dest_img_dir}/src_FA.nii.gz

      # Ensure the variables are parsed into Singularity
      export SINGULARITYENV_dest_img_dir=${dest_img_dir}
      export SINGULARITYENV_deform_dir=${deform_dir}
      export SINGULARITYENV_affine_param_file=${affine_param_file}
      export SINGULARITYENV_dest_def_dir=${dest_def_dir}

      # Run Elastix using Singularity
      singularity exec \
        -B ${dest_img_dir}:${dest_img_dir} \
        -B ${deform_dir}:${deform_dir} \
        -B ${affine_param_file}:${affine_param_file} \
        /home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing/elastix.sif \
        elastix -threads 1 -f ${dest_img_dir}/tgt_FA.nii.gz -m ${dest_img_dir}/src_FA.nii.gz -p ${affine_param_file} -out ${deform_dir}

      # Step 3: Warp the moving image to the target space using Transformix
      singularity exec \
        -B ${dest_img_dir}:${dest_img_dir} \
        -B ${deform_dir}:${deform_dir} \
        -B ${dest_def_dir}:${dest_def_dir} \
        /home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing/elastix.sif \
        transformix -in ${dest_img_dir}/src_FA.nii.gz -tp ${deform_dir}/TransformParameters.0.txt -out ${dest_def_dir}

      # Rename the result image
      mv ${dest_def_dir}/result.nii.gz ${dest_img_dir}/warped_FA.nii.gz

      echo "Processed ${sub_id} ${ses1_id} -> ${ses2_id}"
    done
  done
done
