#!/bin/bash

#SBATCH --job-name=deform
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/generate_deformation_fields_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/generate_deformation_fields_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

# Set the base directories
input_dir="/home/groups/dlmrimnd/jacob/data/MND/warped_input"
output_dir="/home/groups/dlmrimnd/jacob/data/MND/deformation_fields"
singularity_image="/home/groups/dlmrimnd/jacob/files/ants2.4.3.sif"

echo "Starting deformation field generation using ANTs"

# Iterate over each subject directory in the input directory
for sub_dir in ${input_dir}/sub-*; do
    full_id=$(basename ${sub_dir})
    echo "Processing subject: ${full_id}"

    # Define paths to the source and target images
    source_img="${sub_dir}/warped_source_Warped.nii.gz"
    target_img="${sub_dir}/warped_target_Warped.nii.gz"

    # Define output directories
    output_sub_dir="${output_dir}/${full_id}"
    
    # Check if the output directory exists and is non-empty
    if [ -d "${output_sub_dir}" ] && [ "$(ls -A ${output_sub_dir})" ]; then
        echo "Skipping ${full_id} as the output directory is non-empty."
        continue
    fi

    mkdir -p ${output_sub_dir}

    # Debug: Check if the source and target images exist
    if [ ! -f "${source_img}" ]; then
        echo "Source image '${source_img}' does not exist."
        continue
    fi
    if [ ! -f "${target_img}" ]; then
        echo "Target image '${target_img}' does not exist."
        continue
    fi

    # Debug: Output paths being used
    echo "Using source image: ${source_img}"
    echo "Using target image: ${target_img}"

    # Run ANTs registration to generate the deformation field and warped image using antsRegistrationSyNQuick.sh
    singularity exec -B ${input_dir}:/input -B ${output_dir}:/output ${singularity_image} \
    antsRegistrationSyNQuick.sh \
    -d 3 \
    -m /input/${full_id}/warped_source_Warped.nii.gz \
    -f /input/${full_id}/warped_target_Warped.nii.gz \
    -t s \
    -o /output/${full_id}/deformation_

    echo "Finished processing ${full_id}"
done

echo "Deformation field generation job completed."
