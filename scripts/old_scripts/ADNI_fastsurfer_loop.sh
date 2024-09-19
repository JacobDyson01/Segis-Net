#!/bin/bash
#SBATCH --job-name=fastsurf
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/fastsurfer_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/fastsurfer_error_%j.err
#SBATCH --partition=cpu
#SBATCH --time=12:00:00


# Full path to the directory containing your FastSurfer input data
data_dir="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/fastsurfer_input_full_head"

# Full path to the directory where you want to store output
output_dir="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/ADNI_output_images"

# Path to the fastsurfer Singularity image
# fastsurfer_image="/home/groups/dlmrimnd/jacob/files/fastsurfer-latest.sif"

# Path to the FreeSurfer license file
# FS_license="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/freesurfer_license/license.txt"

# Iterate over each patient directory in fastsurfer_input
for patient_dir in ${data_dir}/sub-*; do
    # Extract the patient ID
    patient_id=$(basename ${patient_dir})

    # Iterate over each session directory within the patient directory
    for session_dir in ${patient_dir}/ses-*; do
        session_id=$(basename ${session_dir})

        # Define path to the T1-weighted image file
        t1w_file="${session_dir}/anat/${patient_id}_${session_id}_T1w.nii.gz"

        # Check if the T1-weighted file exists and process it
        if [ -f "${t1w_file}" ]; then
            # Define a subject ID for the T1-weighted file
            subject_id="${patient_id}_${session_id}"

            # Submit the job to SLURM for FastSurfer processing
            sbatch srun_fastsurfer_user.sh "$output_dir" "$t1w_file" "${subject_id}" 
        fi
    done
done
