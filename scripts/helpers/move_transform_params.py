import os
import shutil

# Define the source and target directories
source_dir = "/home/groups/dlmrimnd/jacob/data/warped_input_to_sub"
target_dir = "/home/groups/dlmrimnd/jacob/data/transformParameters"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Walk through the source directory to find all TransformParameters files
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.startswith("TransformParameters_") and file.endswith(".txt"):
            source_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_dir, file)

            # Check if the file already exists in the target directory
            if not os.path.exists(target_file_path):
                # Copy the file if it doesn't exist in the target directory
                shutil.copy2(source_file_path, target_file_path)
                print(f"Copied {file} to {target_dir}")
            else:
                print(f"{file} already exists in {target_dir}, skipping.")

print("Script completed.")
