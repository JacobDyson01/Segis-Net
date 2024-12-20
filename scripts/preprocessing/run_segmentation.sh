#!/bin/bash

#SBATCH --job-name=segment
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/logs/run_segment_ouput_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/logs/run_segment_error_%j.err
#SBATCH --partition=a100
#SBATCH --time=250:00:00



# Load necessary modules
# module load python/3.6.7
# module load tensorflow/1.4.0
# module load keras/2.2.0
# module load scikit-learn/0.19.1

# /home/groups/dlmrimnd/jacob/miniconda3/envs/segisnet_env/bin/python
# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate
source /home/groups/dlmrimnd/jacob/miniconda3/bin/activate segisnet_env

# Run the training script
/home/groups/dlmrimnd/jacob/miniconda3/envs/segisnet_env/bin/python3.6 /home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/segment_MND_reversed.py