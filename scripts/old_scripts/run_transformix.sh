#!/bin/bash

#SBATCH --job-name=elastix_transformix_job
#SBATCH --output=/home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing/logs/output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing/logs/error_%j.err
#SBATCH --partition=cpu

# Load necessary modules if required
# module load necessary_module

# Directory containing input subjects
input_dir="/home/groups/dlmrimnd/jacob/data/input_images"

# Output directory for deformation fields
output_base_dir="/home/groups/dlmrimnd/jacob/data/deformations"
mkdir -p ${output_base_dir}

# Loop through all subject-session folders
for subject_session in ${input_dir}/sub-*; do
  sub_ses_id=$(basename ${subject_session})
  echo "Processing subject-session: ${sub_ses_id}"

  # Define the path for transform parameters file
  transform_params_file="/home/groups/dlmrimnd/jacob/data/Preprocess_affine_transformParam/${sub_ses_id}/TransformParameters.0.txt"

  # Define the output directory for deformation fields
  deform_dir="${output_base_dir}/${sub_ses_id}"
  mkdir -p ${deform_dir}

  # Generate the deformation field using Transformix
  singularity exec \
    -B ${subject_session}:${subject_session} \
    -B ${deform_dir}:${deform_dir} \
    /home/groups/dlmrimnd/jacob/code/segis-net/Elastix_Preprocessing/elastix.sif \
    transformix -def all -tp ${transform_params_file} -out ${deform_dir}

  echo "Processed ${sub_ses_id}"
done