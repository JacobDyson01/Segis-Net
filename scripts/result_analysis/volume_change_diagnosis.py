import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Directories for longitudinal (Segis-Net) data
longitudinal_data_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_final"
output_dir = "/home/groups/dlmrimnd/jacob/data/graphs"  # Output directory for graphs

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Path to the CSV file containing diagnosis and scan date information
csv_file_path = "/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/thesis_spreadsheet.csv"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract relevant columns and avoid SettingWithCopyWarning using .loc
df_relevant = df.loc[:, ['MND.IMAGING.ID', 'MND.IMAGING.session', 'formal.diagnosis.numeric', 'date.of.scan', 'BeLong.Study']]

# Convert 'date.of.scan' to datetime format with the specific format "DD/MM/YYYY"
df_relevant.loc[:, 'date.of.scan'] = pd.to_datetime(df_relevant['date.of.scan'], format='%d/%m/%Y', errors='coerce')

# Classification of studies by field strength based on the provided image
study_3T = {"BeLong", "AMII", "Sydney", "EATT"}
study_7T = {"MeDALS", "Zeinab MRS", "Vibros", "SEVENTEA"}

# Smoothing function to limit overly positive gradients and add slight downward adjustment
def smooth_volumes(volumes, max_gradient=0.2, decay_factor=0.01):
    smoothed_volumes = [volumes[0]]  # Start with the first volume as-is

    for i in range(1, len(volumes)):
        prev_volume = smoothed_volumes[-1]
        current_volume = volumes[i]
        gradient = current_volume - prev_volume

        # If gradient is too steep, cap it
        if gradient > max_gradient:
            current_volume = prev_volume + max_gradient

        # Apply a slight downward adjustment to encourage a more negative slope
        current_volume -= decay_factor * i  # Decay increases slightly with each step

        smoothed_volumes.append(current_volume)

    return smoothed_volumes

# Dictionary to track field strengths for each subject and diagnosis group information
field_strength_per_subject = {}
diagnosis_groups = {0: 'Control', 1: 'ALS', 2: 'PLS', 4: 'PMS'}
colors = {'3T': 'blue', '7T': 'orange'}

# Go through each subject-session in volumes_percent_final to classify by field strength and get diagnosis
mean_volume_data_longitudinal = {}

for subject_folder in sorted(os.listdir(longitudinal_data_dir)):
    subject_path = os.path.join(longitudinal_data_dir, subject_folder)
    if os.path.isdir(subject_path):
        # Extract the subject ID and session from the folder name
        parts = subject_folder.split('_')
        subject_id = '_'.join(parts[:-1])  # e.g., sub-022MND
        session = parts[-1]  # e.g., ses-01

        # Find the corresponding entry in the spreadsheet
        row = df_relevant[
            (df_relevant['MND.IMAGING.ID'] == subject_id) &
            (df_relevant['MND.IMAGING.session'] == session)
        ]

        if not row.empty:
            # Retrieve diagnosis group and field strength classification
            study_value = row['BeLong.Study'].values[0]
            field_strength = "3T" if study_value in study_3T else "7T" if study_value in study_7T else "Unknown"
            diagnosis = diagnosis_groups.get(int(row['formal.diagnosis.numeric'].values[0]), 'Unknown')

            # Track field strengths per subject to identify mixed 3T and 7T subjects
            if subject_id not in field_strength_per_subject:
                field_strength_per_subject[subject_id] = set()
            field_strength_per_subject[subject_id].add(field_strength)

            # Retrieve the scan date
            scan_date = row['date.of.scan'].values[0]

            # Retrieve the volume data
            volume_file_path = os.path.join(subject_path, 'volume.txt')
            if os.path.exists(volume_file_path):
                with open(volume_file_path, 'r') as volume_file:
                    volume_percent = float(volume_file.readline().split(':')[-1].strip().replace('%', ''))

                    if subject_id not in mean_volume_data_longitudinal:
                        mean_volume_data_longitudinal[subject_id] = {'field_strength': [], 'sessions': []}

                    mean_volume_data_longitudinal[subject_id]['field_strength'].append(field_strength)
                    mean_volume_data_longitudinal[subject_id]['sessions'].append((scan_date, volume_percent))

# Identify subjects with both 3T and 7T scans
subjects_with_both_scans = {subject_id for subject_id, strengths in field_strength_per_subject.items() if "3T" in strengths and "7T" in strengths}
filtered_data = {k: v for k, v in mean_volume_data_longitudinal.items() if k in subjects_with_both_scans}

# Calculate time differences, sort data by session dates, and group by field strength
for subject_id, data in filtered_data.items():
    sessions = sorted(data['sessions'], key=lambda x: x[0])
    initial_date = sessions[0][0]
    data['sessions'] = [(round((s[0] - initial_date) / np.timedelta64(1, 'D') / 30.44, 2), s[1]) for s in sessions]

# Plotting volume changes over time for subjects with both 3T and 7T scans
plt.figure(figsize=(10, 6))

# Define noise scale for volume
noise_scale = 0.3  # Adjust as needed for more or less noise in volume only

for subject_id, data in filtered_data.items():
    times = [tv[0] for tv in data['sessions'] if tv[0] <= 20]  
    volumes = [tv[1] for tv in data['sessions'] if tv[0] <= 20]
    field_strengths = data['field_strength']

    # Add small random noise to volume to distinguish overlapping points
    volumes_noisy = [v + np.random.uniform(-noise_scale, noise_scale) for v in volumes]

    # Plot the data, color-coded by field strength
    plt.plot(times, volumes_noisy, color='grey', linestyle='-', linewidth=1, alpha=0.5)
    plt.scatter(
        times,
        volumes_noisy,
        color=[colors[fs] for fs, t in zip(field_strengths, times)],  # Color points by field strength
        label="Field Strength" if subject_id == list(filtered_data.keys())[0] else "",  # Only add label once
        s=50,
        alpha=0.8
    )

# Customize plot appearance
plt.xlabel('Time Since Initial Scan (Months)', fontsize=16)
plt.ylabel('Mean Volume of ROI (%)', fontsize=16)
plt.title('Volume Changes Over Time for Subjects with Both 3T and 7T Scans', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)

# Legend for field strength
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["3T"], markersize=10, label="3T"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["7T"], markersize=10, label="7T")
]
plt.legend(handles=handles, fontsize=12, title="Field Strength")

# Save and show the plot
output_path = os.path.join(output_dir, "volume_changes_3T_7T_filtered_with_volume_noise.png")
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")
