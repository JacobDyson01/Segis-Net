#!/bin/bash

#SBATCH --job-name=fastsurfer_job1 
#SBATCH --ntasks=1 
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/fastsurfer_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/fastsurfer_error_%j.err
#SBATCH --partition=p100                 
#SBATCH --gres=gpu:1                                    
#SBATCH --time=24:00:00 

# Set paths based on your arguments
outputdir="$1"
datadir="$2"
# fastsurfer_image="$3"  # Use the provided argument
sid="$3"
# license="$5"

# data_dir="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts" 
# output_dir="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/ADNI_output_images"
FS_dir="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/freesurfer_license"
# Load the Singularity module (if needed)


# Singularity execution command
# singularity run --nv --no-home \
#   -B "$data_dir" \
#   -B "$output_dir" \
#   -B "$FS_dir" \
#   "$fastsurfer_image" \
#   --fs_license "$license" \
#   --sd "$outputdir" \
#   --t1 "$datadir" \
#   --sid "$sid"
export TMPDIR="/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/temp"
singularity exec --nv --no-home \
                -B /home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/fastsurfer_input_full_head:/input \
                /home/groups/dlmrimnd/jacob/files/fastsurfer-latest.sif \
                ./run_fastsurfer.sh \
                --fs_license /home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/freesurfer_license/license.txt \
                --t1 "$datadir" \
                --sid "$sid" \
                --sd "$outputdir" \
                --py python --fsqsphere