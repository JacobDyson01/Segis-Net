import os
import pandas as pd

# Define the paths
csv_path = '/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/thesis_spreadsheet.csv'
updated_csv_path = '/home/groups/dlmrimnd/jacob/projects/Segis-Net/scripts/thesis_spreadsheet_updated.csv'
volumes_dir = '/home/groups/dlmrimnd/jacob/data/volumes_percent_final/'

# Load the spreadsheet
df = pd.read_csv(csv_path)

# Initialize a dictionary to count studies
study_counts = {}

# Create a new column in the dataframe for storing volume percent
df['VOLUME.PERCENT.ROI'] = None

# Iterate through the directories in volumes_percent_final
for subject_folder in os.listdir(volumes_dir):
    folder_path = os.path.join(volumes_dir, subject_folder)

    # Parse the subject ID and session from the folder name (e.g., 'sub-002MND_ses-01')
    if os.path.isdir(folder_path):
        try:
            subject_id, session_id = subject_folder.split('_ses-')
            session_id = f'ses-{session_id}'  # Ensure correct session format

            # Debug information for every subject and session being processed
            print(f"Processing {subject_id}, {session_id}")

            # Check if subject_id and session_id exist in the spreadsheet
            matched_row = df[(df['MND.IMAGING.ID'] == subject_id) & (df['MND.IMAGING.session'] == session_id)]
            
            # Debug: Check if matched_row is empty or not
            if matched_row.empty:
                print(f"No match found for {subject_id} {session_id} in spreadsheet.")
                continue  # Skip to the next subject/session if no match
            else:
                print(f"Match found for {subject_id} {session_id} in spreadsheet.")

            # Get the BeLong Study value
            study_name = matched_row['BeLong.Study'].values[0]
            print(f"Study Name: {study_name}")

            # Update the study counts
            if study_name in study_counts:
                study_counts[study_name] += 1
            else:
                study_counts[study_name] = 1

            # Read the volume.txt file and update the spreadsheet
            volume_file = os.path.join(folder_path, 'volume.txt')
            if os.path.exists(volume_file):
                with open(volume_file, 'r') as f:
                    volume_text = f.read().strip()

                    # Debug: Show content of volume.txt
                    print(f"Volume file contents for {subject_id} {session_id}: {volume_text}")

                    # Extract the numeric value from the text
                    # Handle both 'Source Volume' and 'Target Volume' cases
                    if 'Source Volume as percentage of eTIV:' in volume_text:
                        volume_value = volume_text.split(':')[-1].strip().rstrip('%')
                    elif 'Target Volume as percentage of eTIV:' in volume_text:
                        volume_value = volume_text.split(':')[-1].strip().rstrip('%')
                    else:
                        print(f"Volume text format not recognized for {subject_id} {session_id}. Skipping...")
                        continue

                    # Convert volume to float
                    try:
                        volume_value = float(volume_value)
                        print(f"Volume Value: {volume_value}")  # Debug: Show volume value being processed
                    except ValueError:
                        print(f"Could not convert volume value for {subject_id} {session_id}: {volume_text}")
                        continue

                    # Update the VOLUME.PERCENT.ROI column in the spreadsheet
                    df.loc[(df['MND.IMAGING.ID'] == subject_id) & (df['MND.IMAGING.session'] == session_id), 'VOLUME.PERCENT.ROI'] = volume_value
                    print(f"Updated CSV for {subject_id} {session_id} with volume {volume_value}")  # Debug: Indicate update success
            else:
                print(f"Volume file not found for {subject_id} {session_id}")

        except Exception as e:
            print(f"Error processing {subject_folder}: {str(e)}")

# Print the study counts
print("Study Counts:")
for study, count in study_counts.items():
    print(f"{study}: {count}")

# Save the updated spreadsheet to a new file
df.to_csv(updated_csv_path, index=False)
print(f"Updated spreadsheet saved to {updated_csv_path}")
