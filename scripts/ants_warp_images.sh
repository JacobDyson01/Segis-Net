#!/bin/bash
#SBATCH --job-name=ants_registration
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/ants_registration_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/ants_registration_error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

# Set the base directories
input_dir="/home/groups/dlmrimnd/jacob/data/MND/input_images"
output_dir="/home/groups/dlmrimnd/jacob/data/MND/ants_warped_input"
template_img="/home/groups/dlmrimnd/jacob/data/brain_mni.nii.gz"
transformation_dir="/home/groups/dlmrimnd/jacob/data/MND/transformation_matrices"
singularity_image="/home/groups/dlmrimnd/jacob/files/ants2.4.3.sif"

echo "Starting ANTs registration job using Singularity"

# Iterate over each subject directory in the input directory
for sub_dir in ${input_dir}/sub-*; do
    full_id=$(basename ${sub_dir})
    echo "Processing subject: ${full_id}"

    # Extract session IDs from full_id (assuming it's of the form sub-xxx_ses-01_ses-02)
    IFS='_' read -r -a id_parts <<< "${full_id}"
    sub_id=${id_parts[0]}
    ses1_id=${id_parts[1]}
    ses2_id=${id_parts[2]}

    # Define paths to the source and target images
    source_img="${sub_dir}/source.nii.gz"
    target_img="${sub_dir}/target.nii.gz"

    # Define output directories
    warped_output_dir="${output_dir}/${full_id}"
    mkdir -p ${warped_output_dir}

    # Define transformation matrix directories for each session
    source_transform_dir="${transformation_dir}/${sub_id}_${ses1_id}"
    target_transform_dir="${transformation_dir}/${sub_id}_${ses2_id}"
    mkdir -p ${source_transform_dir}
    mkdir -p ${target_transform_dir}

    # Run ANTs registration for source image
    singularity exec -B ${input_dir}:/input -B ${output_dir}:/output -B ${template_img}:/template/brain_mni.nii.gz -B ${transformation_dir}:/transformation ${singularity_image} \
    antsRegistrationSyNQuick.sh \
    -d 3 \
    -m /input/${full_id}/source.nii.gz \
    -f /template/brain_mni.nii.gz \
    -t r \
    -o /output/${full_id}/source_

    # Save the affine matrix for source image
    cp /home/groups/dlmrimnd/jacob/data/ants_warped_input/${full_id}/source_0GenericAffine.mat ${source_transform_dir}/affine.mat

    # Run ANTs registration for target image
    singularity exec -B ${input_dir}:/input -B ${output_dir}:/output -B ${template_img}:/template/brain_mni.nii.gz -B ${transformation_dir}:/transformation ${singularity_image} \
    antsRegistrationSyNQuick.sh \
    -d 3 \
    -m /input/${full_id}/target.nii.gz \
    -f /template/brain_mni.nii.gz \
    -t r \
    -o /output/${full_id}/target_

    # Save the affine matrix for target image
    cp /home/groups/dlmrimnd/jacob/data/ants_warped_input/${full_id}/target_0GenericAffine.mat ${target_transform_dir}/affine.mat

    echo "Finished processing ${full_id}"
done

echo "ANTs registration job completed."
