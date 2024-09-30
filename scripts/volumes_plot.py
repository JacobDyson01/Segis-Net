import os
import matplotlib.pyplot as plt
import numpy as np

# Directories to iterate through
data_dir = "/home/groups/dlmrimnd/jacob/data/volumes_ADNI"
output_dir = "/home/groups/dlmrimnd/jacob/data/graphs"  # Output directory for graphs

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Dictionary to hold subject IDs and their corresponding session volumes
subject_data = {}

# Volume thresholds
# lower_bound_threshold = 10000
# upper_bound_threshold = 50000
lower_bound_threshold = 0
upper_bound_threshold = 500000

# Traverse the directory
for subject_folder in sorted(os.listdir(data_dir)):
    subject_path = os.path.join(data_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        # Extract subject ID and session number from folder name (e.g., sub-xxx_ses_02)
        subject_id = '_'.join(subject_folder.split('_')[:-1])  # Extract subject id (e.g., sub-xxx)
        session_part = subject_folder.split('_')[-1]  # Get the part like "ses_02"
        session_number = int(session_part.split('_')[-1])  # Extract the number (e.g., 02 -> 2)

        # Only consider sessions 1, 2, 3, and 4
        if session_number <= 4:
            # Path to the volume.txt file
            volume_file_path = os.path.join(subject_path, 'volume.txt')

            if os.path.exists(volume_file_path):
                # Read the volume value
                with open(volume_file_path, 'r') as volume_file:
                    line = volume_file.readline().strip()
                    volume_value = float(line.split(':')[-1].strip().split()[0])  # Assuming "Total volume: <value>"

                    # Apply lower and upper bound thresholds
                    if lower_bound_threshold <= volume_value <= upper_bound_threshold:
                        # Store the volume in the subject's session data
                        if subject_id not in subject_data:
                            subject_data[subject_id] = {}
                        subject_data[subject_id][session_number] = volume_value
                    else:
                        print(f"Excluding outlier volume {volume_value} in {subject_folder}")
            else:
                print(f"Warning: volume.txt not found in {subject_folder}")

# Plot the individual volume values for each subject, connecting them with lines
plt.figure(figsize=(10, 6))

# Plot each subject's volume across sessions
for subject_id, session_volumes in subject_data.items():
    # Ensure the subject has data for sessions 1-4 (can handle missing sessions)
    session_numbers = sorted(session_volumes.keys())
    volumes = [session_volumes[sn] for sn in session_numbers]
    
    # Plot individual subject data points and connect them with a line
    plt.plot(session_numbers, volumes, marker='o', linestyle='-', color='gray', alpha=0.5)

# Calculate mean volume for each session (only for sessions that have data)
mean_volumes = [np.mean([v[sn] for v in subject_data.values() if sn in v]) for sn in range(1, 5)]

# Plot the mean volume for each session and connect them with a red line
mean_session_numbers = [1, 2, 3, 4]
plt.plot(mean_session_numbers, mean_volumes, marker='o', linestyle='-', color='r', label="Mean Volume", linewidth=2)

# Adding labels and title
plt.xlabel('Session Number')
plt.ylabel('Volume (mm^3)')
plt.title('Volume vs Session Number FastSurfer ADNI NC')
plt.grid(True)
plt.legend()

# Save the plot as a PNG file
output_path = os.path.join(output_dir, "volume_vs_session_ADNI_subject_lines_fastsurfer.png")
plt.savefig(output_path)  # Saves the plot as a PNG

# Show the plot
plt.show()

print(f"Plot saved to {output_path}")
