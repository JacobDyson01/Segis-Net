#!/bin/bash

# Set directories
data_dir="/home/groups/dlmrimnd/jacob/data/volumes_MND"
new_input_dir="/home/groups/dlmrimnd/jacob/data/MND_volumes"
mkdir -p ${new_input_dir}
# Loop through each subject directory
for subject_dir in ${data_dir}/sub-*; do
    # Extract the subject ID by cutting the directory name before the first underscore
    subject_id=$(basename ${subject_dir} | cut -d'_' -f1)
    
    # Find all session directories for this subject
    sessions=($(find ${data_dir}/${subject_id}_* -maxdepth 1 -type d -name "*ses-*"))

    # Loop over all session combinations, keeping chronological order
    for ((i=0; i<${#sessions[@]}-1; i++)); do
        for ((j=i+1; j<${#sessions[@]}; j++)); do
            ses1_dir=${sessions[$i]}
            ses2_dir=${sessions[$j]}

            # Extract session information (e.g., ses-01, ses-02) and replace dash with an underscore
            ses1=$(basename ${ses1_dir} | grep -o "ses-[0-9]*" | sed 's/-/_/g')  
            ses2=$(basename ${ses2_dir} | grep -o "ses-[0-9]*" | sed 's/-/_/g')  

            # Create the new directory for the paired sessions
            new_sub_dir="${new_input_dir}/${subject_id}_${ses1}_${ses2}"
            mkdir -p ${new_sub_dir}
            
            # Copy the volume.txt file from ses1 and ses2, renaming them
            if [ -f "${ses1_dir}/volume.txt" ]; then
                cp "${ses1_dir}/volume.txt" "${new_sub_dir}/source.txt"
                echo "Copied ${ses1_dir}/volume.txt to ${new_sub_dir}/source.txt"
            else
                echo "Warning: volume.txt not found in ${ses1_dir}"
            fi

            if [ -f "${ses2_dir}/volume.txt" ]; then
                cp "${ses2_dir}/volume.txt" "${new_sub_dir}/target.txt"
                echo "Copied ${ses2_dir}/volume.txt to ${new_sub_dir}/target.txt"
            else
                echo "Warning: volume.txt not found in ${ses2_dir}"
            fi

            echo "Created directory ${new_sub_dir} for paired sessions ${ses1} and ${ses2}."
        done
    done
done
