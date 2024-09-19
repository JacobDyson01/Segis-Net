#!/bin/bash

#SBATCH --job-name=elastix_job
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/output_%A_%a.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/logs/error_%A_%a.err
#SBATCH --array=0-113 # Adjust the range based on the number of lines in the CSV file minus one change to 0-113
#SBATCH --partition=cpu



# Load necessary modules if required
# module load necessary_module
# module load elastix/5.0.1
# Read the CSV file
CSV_FILE="/home/groups/dlmrimnd/akshit/ADNI_NC_filenames.csv"
filenames=($(cat $CSV_FILE)) 

# Get the subject name for this task
SUBJNAME=${filenames[$SLURM_ARRAY_TASK_ID]}

# Call the elastix.sh script with the subject name
/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/elastix_def_only.sh ${SUBJNAME}
