import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Directories for longitudinal (Segis-Net) data
longitudinal_data_dir = "/home/groups/dlmrimnd/jacob/data/volumes_percent_final"
output_dir = "/home/groups/dlmrimnd/jacob/data/graphs"  # Output directory for graphs

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Path to the CSV file containing diagnosis information
csv_file_path = "/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/thesis_spreadsheet.csv"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract relevant columns: 'MND.IMAGING.ID', 'MND.IMAGING.session', 'formal.diagnosis.numeric'
df_relevant = df[['MND.IMAGING.ID', 'MND.IMAGING.session', 'formal.diagnosis.numeric']]

def calculate_percent_changes(subject_data):
    percent_change_data = {}
    for subject_id, session_volumes in subject_data.items():
        session_numbers = sorted(session_volumes.keys())
        if len(session_numbers) > 1:  # Need at least two sessions to calculate a change
            percent_changes = []
            for i in range(1, len(session_numbers)):
                session1 = session_volumes[session_numbers[i-1]]
                session2 = session_volumes[session_numbers[i]]
                percent_change = ((session2 - session1) / session1) * 100  # Calculate percent change
                percent_changes.append(percent_change)
            percent_change_data[subject_id] = percent_changes
    return percent_change_data

def get_volume_data(data_dir):
    subject_data = {}
    for subject_folder in sorted(os.listdir(data_dir)):
        subject_path = os.path.join(data_dir, subject_folder)
        if os.path.isdir(subject_path):
            # Extract subject ID and session number
            subject_id = '_'.join(subject_folder.split('_')[:-1])
            session_number = int(subject_folder.split('_')[-1].split('-')[-1])

            volume_file_path = os.path.join(subject_path, 'volume.txt')
            if os.path.exists(volume_file_path):
                with open(volume_file_path, 'r') as volume_file:
                    line = volume_file.readline().strip()
                    volume_percent = float(line.split(':')[-1].strip().replace('%', ''))

                    if subject_id not in subject_data:
                        subject_data[subject_id] = {}
                    subject_data[subject_id][session_number] = volume_percent
    return subject_data

# Get the volume data for longitudinal (Segis-Net)
subject_data_longitudinal = get_volume_data(longitudinal_data_dir)

# Calculate the percent changes for longitudinal data
percent_change_data_longitudinal = calculate_percent_changes(subject_data_longitudinal)

# Adding the diagnosis information
def get_diagnosis_group(subject_id, session_number):
    # Find the diagnosis for the specific subject and session
    row = df_relevant[
        (df_relevant['MND.IMAGING.ID'] == subject_id) &
        (df_relevant['MND.IMAGING.session'] == f"ses-{session_number:02d}")
    ]
    if not row.empty:
        return row['formal.diagnosis.numeric'].values[0]
    return None

# Now group the longitudinal data by diagnosis and calculate the mean/std for each group
diagnosis_groups = {}

for subject_id, changes in percent_change_data_longitudinal.items():
    for session_number, percent_change in enumerate(changes, start=1):
        diagnosis = get_diagnosis_group(subject_id, session_number)
        if diagnosis is not None:
            if diagnosis not in diagnosis_groups:
                diagnosis_groups[diagnosis] = []
            diagnosis_groups[diagnosis].append(percent_change)

# Plot the results
plt.figure(figsize=(10, 6))

# Iterate over each diagnosis group and plot the longitudinal data
colors = ['orange', 'blue', 'green', 'red']
for idx, (diagnosis, changes) in enumerate(diagnosis_groups.items()):
    mean_percent_changes = [np.mean(changes)] * len(changes)
    std_percent_changes = [np.std(changes)] * len(changes)
    
    # Session transition numbers (1→2, 2→3, etc.)
    session_change_numbers = range(1, len(changes) + 1)
    
    # Plot longitudinal data for this diagnosis group
    plt.errorbar(session_change_numbers, changes, yerr=std_percent_changes, 
                 fmt='o-', color=colors[idx], ecolor='lightgray', elinewidth=2, capsize=4,
                 label=f'Diagnosis Group {diagnosis}')

# Adjusting the font size for axes, title, ticks, and legend
plt.xlabel('Session Change (01→02, 02→03, 03→04)', fontsize=16)  # Larger x-axis label font
plt.ylabel('Percentage Change in Volume of ROI', fontsize=16)  # Larger y-axis label font
plt.title('Mean Percentage Change in Volume of ROI by Diagnosis Group', fontsize=15)  # Larger title font
plt.xticks([1, 2, 3], ['01→02', '02→03', '03→04'], fontsize=14)  # Larger tick font size for x-axis
plt.yticks(fontsize=14)  # Larger tick font size for y-axis

# Show legend with a larger font
plt.legend(fontsize=14)

# Show grid
plt.grid(True)

# Save the plot as a PNG file
output_path = os.path.join(output_dir, "percentage_change_vs_diagnosis_longitudinal.png")
plt.savefig(output_path)  # Saves the plot as a PNG

# Show the plot
plt.show()

print(f"Plot saved to {output_path}")
