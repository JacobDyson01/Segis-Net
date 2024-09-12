#!/bin/bash

#SBATCH --job-name=preprocess_deformation_fields
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/preprocess_deformation_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/preprocess_deformation_%j.err
#SBATCH --partition=cpu
#SBATCH --time=24:00:00

# Constants for paths
data_dir="/home/groups/dlmrimnd/akshit/MND_output_files"
new_input_dir="/home/groups/dlmrimnd/jacob/data/MND_input_images"
output_dir="/home/groups/dlmrimnd/jacob/data/MND_warped_input"
template_img="/home/groups/dlmrimnd/jacob/data/brain_mni.nii.gz"
singularity_image="/home/groups/dlmrimnd/jacob/files/ants2.4.3.sif"

# Create new input directory if it doesn't exist
mkdir -p ${new_input_dir}

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

            ses1=$(basename ${ses1_dir}) # e.g., ses-01
            ses2=$(basename ${ses2_dir}) # e.g., ses-02
            
            # Define new input folder using the correct format: sub-xxxMND_ses-01_ses-02
            new_sub_dir="${new_input_dir}/${subject_id}_${ses1}_${ses2}"
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

            # Run ANTs registration for source -> template
            echo "Warping source image to template for ${subject_id}_${ses1}_${ses2}"
            singularity exec -B ${new_input_dir}:/input -B ${output_dir}:/output -B ${template_img}:/template ${singularity_image} \
            antsRegistrationSyNQuick.sh \
            -d 3 \
            -m /input/${subject_id}_${ses1}_${ses2}/source.nii.gz \
            -f /template \
            -t s \
            -o /output/${subject_id}_${ses1}_${ses2}/warped_source_

            # Run ANTs registration for target -> template
            echo "Warping target image to template for ${subject_id}_${ses1}_${ses2}"
            singularity exec -B ${new_input_dir}:/input -B ${output_dir}:/output -B ${template_img}:/template ${singularity_image} \
            antsRegistrationSyNQuick.sh \
            -d 3 \
            -m /input/${subject_id}_${ses1}_${ses2}/target.nii.gz \
            -f /template \
            -t s \
            -o /output/${subject_id}_${ses1}_${ses2}/warped_target_

        done
    done
done

echo "Preprocessing and deformation field generation complete!"
