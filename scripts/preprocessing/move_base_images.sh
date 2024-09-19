#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/preprocess_deformation_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/preprocess_deformation_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

# Constants for paths
data_dir="/home/groups/dlmrimnd/akshit/MND_output_files"
new_input_dir="/home/groups/dlmrimnd/jacob/data/MND/input_images"
output_dir="/home/groups/dlmrimnd/jacob/data/MND/warped_input"
template_img="/home/groups/dlmrimnd/jacob/data/brain_mni.nii.gz"
transformation_dir="/home/groups/dlmrimnd/jacob/data/MND/transformation_matrices"
binary_mask_dir="/home/groups/dlmrimnd/jacob/data/MND/binary_masks"
cropped_input_dir="/home/groups/dlmrimnd/jacob/data/MND/cropped_input"
cropped_masks_dir="/home/groups/dlmrimnd/jacob/data/MND/cropped_masks"
singularity_image="/home/groups/dlmrimnd/jacob/files/ants2.4.3.sif"

# Remove existing directories if they exist to avoid overlap
if [ -d "${new_input_dir}" ]; then
    echo "Removing existing directory ${new_input_dir}"
    rm -rf ${new_input_dir}
fi
if [ -d "${output_dir}" ]; then
    echo "Removing existing directory ${output_dir}"
    rm -rf ${output_dir}
fi

# Create new input and output directories
mkdir -p ${new_input_dir}
mkdir -p ${output_dir}

echo "Starting the preprocessing pipeline and deformation field generation using ANTs"

# Iterate over each subject in the data directory
for subject_dir in ${data_dir}/sub-*; do
    subject_id=$(basename ${subject_dir} | cut -d'_' -f1)
    
    # Gather all session directories for this subject
    sessions=($(find ${data_dir}/${subject_id}_* -maxdepth 1 -type d -name "*ses-*"))

    # If the subject has only one session, skip it
    if [ ${#sessions[@]} -le 1 ]; then
        echo "Skipping ${subject_id} as it has only one session."
        continue
    fi

    # Loop over all session combinations while keeping chronological order
    for ((i=0; i<${#sessions[@]}-1; i++)); do
        for ((j=i+1; j<${#sessions[@]}; j++)); do
            ses1_dir=${sessions[$i]}
            ses2_dir=${sessions[$j]}

            # Extract session part (ses-01, ses-02) from directory name
            ses1=$(basename ${ses1_dir} | grep -o "ses-[0-9]*" | sed 's/-/_/g')  # Extract ses-01 and replace dash with underscore
            ses2=$(basename ${ses2_dir} | grep -o "ses-[0-9]*" | sed 's/-/_/g')  # Extract ses-02 and replace dash with underscore
            
            # Define new input folder using the correct format: sub-xxxMND_ses_01_ses_02
            new_sub_dir="${new_input_dir}/${subject_id}_${ses1}_${ses2}"  # Correct folder structure
            mkdir -p ${new_sub_dir}
            
            # Define the source and target .mgz file paths
            source_img="${ses1_dir}/mri/brain.mgz"
            target_img="${ses2_dir}/mri/brain.mgz"

            # Call Python script to convert .mgz to .nii.gz
            python3 convert_mgz_to_nii.py ${source_img} ${new_sub_dir}/source.nii.gz
            python3 convert_mgz_to_nii.py ${target_img} ${new_sub_dir}/target.nii.gz
            
            echo "Created input files for ${subject_id}_${ses1}_${ses2}"

            # Create output directory for this subject/session pair
            warped_output_dir="${output_dir}/${subject_id}_${ses1}_${ses2}"
            mkdir -p ${warped_output_dir}

            # Define transformation matrix directories for each session
            source_transform_dir="${transformation_dir}/${subject_id}_${ses1}"
            target_transform_dir="${transformation_dir}/${subject_id}_${ses2}"
            mkdir -p ${source_transform_dir}
            mkdir -p ${target_transform_dir}

            # Run ANTs registration for source -> template
            echo "Warping source image to template for ${subject_id}_${ses1}_${ses2}"
            singularity exec -B ${new_input_dir}:/input -B ${output_dir}:/output -B /home/groups/dlmrimnd/jacob/data:/data ${singularity_image} \
            antsRegistrationSyNQuick.sh \
            -d 3 \
            -m /input/${subject_id}_${ses1}_${ses2}/source.nii.gz \
            -f /data/brain_mni.nii.gz \
            -t r \
            -o /output/${subject_id}_${ses1}_${ses2}/warped_source_

            # Save the affine matrix for source image
            cp ${output_dir}/${subject_id}_${ses1}_${ses2}/warped_source_0GenericAffine.mat ${source_transform_dir}/affine.mat

            # Run ANTs registration for target -> template
            echo "Warping target image to template for ${subject_id}_${ses1}_${ses2}"
            singularity exec -B ${new_input_dir}:/input -B ${output_dir}:/output -B /home/groups/dlmrimnd/jacob/data:/data ${singularity_image} \
            antsRegistrationSyNQuick.sh \
            -d 3 \
            -m /input/${subject_id}_${ses1}_${ses2}/target.nii.gz \
            -f /data/brain_mni.nii.gz \
            -t r \
            -o /output/${subject_id}_${ses1}_${ses2}/warped_target_

            # Save the affine matrix for target image
            cp ${output_dir}/${subject_id}_${ses1}_${ses2}/warped_target_0GenericAffine.mat ${target_transform_dir}/affine.mat

        done
    done
done