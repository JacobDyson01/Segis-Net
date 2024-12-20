#!/bin/bash

#SBATCH --job-name=combined_segis
#SBATCH --output=/home/groups/dlmrimnd/jacob/projects/Segis-Net/logs/upgraded_segis_output_%j.out
#SBATCH --error=/home/groups/dlmrimnd/jacob/projects/Segis-Net/logs/upgraded_segis_error_%j.err
#SBATCH --partition=a100
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1



# Load necessary modules
# module load python/3.6.7
# module load tensorflow/1.4.0
# module load keras/2.2.0
# module load scikit-learn/0.19.1

# /home/groups/dlmrimnd/jacob/miniconda3/envs/segisnet_env/bin/python
# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate
module load cuda/11.4
source /home/groups/dlmrimnd/jacob/miniconda3/bin/activate segis-new-env

export PYTHONUNBUFFERED=1

/home/groups/dlmrimnd/jacob/miniconda3/envs/upgraded-segis-env/bin/python3.9 /home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/gpu_test.py


# Run the training script
/home/groups/dlmrimnd/jacob/miniconda3/envs/upgraded-segis-env/bin/python3.9 /home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/Train_SegisNet.py