#!/bin/bash

# Base directory where the images are stored
base_dir="/home/groups/dlmrimnd/jacob/data/fastsurfer_input"

# Find all .nii.gz files within the base directory and change permissions
find $base_dir -type f -name "*.nii.gz" -exec chmod 640 {} \;

echo "Permissions updated for all .nii.gz files to allow group read access."