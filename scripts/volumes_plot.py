import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

# Directories to iterate through
data_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_MND"
output_dir = "/home/groups/dlmrimnd/jacob/data/graphs"  # Output directory for graphs

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Dictionary to hold subject IDs and their corresponding session volumes
subject_data = {}

# Volume thresholds (since we're plotting percentages, let's remove the threshold unless needed)
lower_bound_threshold = 0
upper_bound_threshold = 100  # Percentage is from 0 to 100, since we're dealing with percentages

# Traverse the directory
for subject_folder in sorted(os.listdir(data_dir)):
    subject_path = os.path.join(data_dir, subject_folder)
    
    if os.path.isdir(subject_path):
        # Extract subject ID and session number from folder name (e.g., sub-xxx_ses-01)
        subject_id = '_'.join(subject_folder.split('_')[:-1])  # Extract subject id (e.g., sub-xxx)
        session_part = subject_folder.split('_')[-1]  # Get the part like "ses-01"
        session_number = int(session_part.split('-')[-1])  # Extract the numeric part (e.g., "01" -> 1)

        # Only consider sessions 1, 2, 3, and 4
        if session_number <= 4:
            # Path to the volume.txt file
            volume_file_path = os.path.join(subject_path, 'volume.txt')

            if os.path.exists(volume_file_path):
                # Read the percentage volume value
                with open(volume_file_path, 'r') as volume_file:
                    line = volume_file.readline().strip()
                    volume_percent = float(line.split(':')[-1].strip().replace('%', ''))  # Get the percentage value

                    # Apply lower and upper bound thresholds
                    if lower_bound_threshold <= volume_percent <= upper_bound_threshold:
                        # Store the percentage volume in the subject's session data
                        if subject_id not in subject_data:
                            subject_data[subject_id] = {}
                        subject_data[subject_id][session_number] = volume_percent
                    else:
                        print(f"Excluding outlier volume {volume_percent}% in {subject_folder}")
            else:
                print(f"Warning: volume.txt not found in {subject_folder}")

# Plot the individual percentage volume values for each subject, connecting them with lines
plt.figure(figsize=(10, 6))

# Create a color map to assign unique colors to each subject
cmap = get_cmap('tab10')  # 'tab10' provides 10 unique colors
colors = [cmap(i % 10) for i in range(len(subject_data))]  # Cycle through colors if more than 10 subjects

# Plot each subject's volume across sessions with different colors
for i, (subject_id, session_volumes) in enumerate(subject_data.items()):
    # Ensure the subject has data for sessions 1-4 (can handle missing sessions)
    session_numbers = sorted(session_volumes.keys())
    volumes = [session_volumes[sn] for sn in session_numbers]
    
    # Plot individual subject data points and connect them with a line
    plt.plot(session_numbers, volumes, marker='o', linestyle='-', color=colors[i], alpha=0.7)

# Calculate mean percentage volume for each session (only for sessions that have data)
mean_volumes = [np.mean([v[sn] for v in subject_data.values() if sn in v]) for sn in range(1, 5)]

# Plot the mean percentage volume for each session and connect them with a black line
mean_session_numbers = [1, 2, 3, 4]
plt.plot(mean_session_numbers, mean_volumes, marker='o', linestyle='-', color='black', linewidth=2)

# Adding labels and title
plt.xlabel('Session Number')
plt.ylabel('Percentage Volume of ROI')
plt.title('Percentage of Volume of ROI in FastSurfer Segmentation (BeLong)')
plt.grid(True)

# Set x-axis to only show 1, 2, 3, 4 without anything in between
plt.xticks([1, 2, 3, 4])

# Remove the legend
plt.legend().remove()

# Save the plot as a PNG file
output_path = os.path.join(output_dir, "percentage_volume_vs_session_MND_volumes_fastsurfer_no_legend.png")
plt.savefig(output_path)  # Saves the plot as a PNG

# Show the plot
plt.show()

print(f"Plot saved to {output_path}")
