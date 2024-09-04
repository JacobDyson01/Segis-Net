#!/bin/bash
#SBATCH --job-name=apply_transform_generic
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/apply_transform_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/apply_transform_error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

# Set the base directories
mask_dir="/home/groups/dlmrimnd/jacob/data/binary_masks"
transform_dir="/home/groups/dlmrimnd/jacob/data/transformation_matrices"
output_dir="/home/groups/dlmrimnd/jacob/data/warped_binary_masks"
singularity_image="/home/groups/dlmrimnd/jacob/files/ants2.4.3.sif"
template_img="/home/groups/dlmrimnd/jacob/data/brain_mni.nii.gz"

echo "Starting mask transformation using ANTs"

# Iterate over each subject directory in the mask directory
for mask_sub_dir in ${mask_dir}/sub-*; do
    sub_id=$(basename ${mask_sub_dir})
    echo "Processing subject: ${sub_id}"

    # Define the paths to the binary mask and transformation matrix
    mask_img="${mask_sub_dir}/binary_mask.nii.gz"
    transform_mat="${transform_dir}/${sub_id}/affine.mat"
    
    # Define the output directory for the warped mask
    output_mask_dir="${output_dir}/${sub_id}/"
    mkdir -p ${output_mask_dir}

    # Apply the transformation to the binary mask using the corresponding affine matrix
    if [ -f "${mask_img}" ] && [ -f "${transform_mat}" ]; then
        singularity exec -B ${mask_dir}:/input -B ${output_dir}:/output -B ${template_img}:/template/brain_mni.nii.gz -B ${transform_dir}:/transforms ${singularity_image} \
        antsApplyTransforms \
        -d 3 \
        -i /input/${sub_id}/binary_mask.nii.gz \
        -r /template/brain_mni.nii.gz \
        -n GenericLabel \
        -o /output/${sub_id}/binary_mask_warped.nii.gz \
        -t /transforms/${sub_id}/affine.mat
    else
        echo "Binary mask or affine matrix not found for ${sub_id}"
    fi

    echo "Finished processing subject: ${sub_id}"
done

echo "Mask transformation job completed."
