#!/bin/bash

# Set the base directory containing the binary_masks subfolders
mask_dir="/home/groups/dlmrimnd/jacob/data/MND/binary_masks"

# Iterate over each subfolder in the binary_masks directory
for folder in "${mask_dir}"/sub-*; do
    # Extract the folder name
    folder_name=$(basename "$folder")

    # Use 'tr' to remove any carriage return characters ('\r')
    cleaned_name=$(echo "$folder_name" | tr -d '\r')

    # Use 'sed' to replace underscores for consistent naming, ensuring proper format
    fixed_name=$(echo "$cleaned_name" | sed 's/ses-/ses_/g')

    # Rename the folder if necessary
    if [ "$folder_name" != "$fixed_name" ]; then
        echo "Renaming: $folder_name -> $fixed_name"
        mv "$folder" "${mask_dir}/$fixed_name"
    fi
done

echo "Renaming complete!"
