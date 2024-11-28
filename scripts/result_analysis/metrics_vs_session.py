import os
import nibabel as nib
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd

# Directories
predicted_dir = '/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_final'
ground_truth_dir = '/home/groups/dlmrimnd/jacob/data/MND/warped_binary_masks_roi'
save_dir = '/home/groups/dlmrimnd/jacob/data/graphs/metrics'

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize a dictionary to store results for each session
session_metrics = {
    'session': [],
    'precision': [],
    'recall': [],
    'dice': []
}

# Function to compute metrics
def compute_metrics(ground_truth, predicted):
    # Flatten the 3D volumes to 1D arrays
    ground_truth_flat = ground_truth.flatten()
    predicted_flat = predicted.flatten()

    # Compute precision, recall, and Dice coefficient
    precision = precision_score(ground_truth_flat, predicted_flat)
    recall = recall_score(ground_truth_flat, predicted_flat)
    
    # Compute Dice coefficient
    intersection = np.sum(predicted_flat[ground_truth_flat == 1])
    dice = (2. * intersection) / (np.sum(ground_truth_flat) + np.sum(predicted_flat))

    return precision, recall, dice

# Define the sessions to consider (first four sessions only)
valid_sessions = ['ses-01', 'ses-02', 'ses-03', 'ses-04']

# Iterate through the predicted segmentations directory
for subject_folder in os.listdir(predicted_dir):
    subject_path = os.path.join(predicted_dir, subject_folder)

    if os.path.isdir(subject_path):
        # Load the source predicted segmentation
        source_predicted_file = os.path.join(subject_path, 'source_segmentation.nii.gz')
        source_predicted_img = nib.load(source_predicted_file)
        source_predicted_data = source_predicted_img.get_fdata()

        # Extract the subject ID and session info from the folder name
        # Example folder name: 'sub-002MND_ses-01_ses-02'
        parts = subject_folder.split('_ses-')
        subject_id = parts[0]  # 'sub-002MND'
        session_ids = [f'ses-{part}' for part in parts[1:] if f'ses-{part}' in valid_sessions]

        # Iterate through the valid sessions (first four sessions)
        for session_id in session_ids:
            # Load the ground truth for the current session
            ground_truth_file = os.path.join(
                ground_truth_dir,
                f'{subject_id}_{session_id}',
                'binary_mask_warped_roi.nii.gz'
            )

            if os.path.exists(ground_truth_file):
                # Load the ground truth binary mask
                ground_truth_img = nib.load(ground_truth_file)
                ground_truth_data = ground_truth_img.get_fdata()

                # Compute metrics for the current session's segmentation
                precision, recall, dice = compute_metrics(ground_truth_data, source_predicted_data)

                # Store results for this session
                session_metrics['session'].append(session_id)
                session_metrics['precision'].append(precision)
                session_metrics['recall'].append(recall)
                session_metrics['dice'].append(dice)

# Convert dictionary to DataFrame for further processing
df_metrics = pd.DataFrame(session_metrics)

# Ensure that session IDs are sorted properly
df_metrics['session'] = pd.Categorical(df_metrics['session'], ordered=True, categories=sorted(valid_sessions))
df_metrics.sort_values('session', inplace=True)

# Compute the average Dice coefficient, precision, and recall for each session
df_avg_metrics = df_metrics.groupby('session').mean().reset_index()

# Plot the average metrics over time (for each session)
plt.figure(figsize=(10, 6))

# Plot Dice coefficient
plt.plot(df_avg_metrics['session'], df_avg_metrics['dice'], label='Average Dice', marker='o')

# Plot Precision
plt.plot(df_avg_metrics['session'], df_avg_metrics['precision'], label='Average Precision', marker='o')

# Plot Recall
plt.plot(df_avg_metrics['session'], df_avg_metrics['recall'], label='Average Recall', marker='o')

# Add labels and legend
plt.xlabel('Session')
plt.ylabel('Average Metric Value')
plt.title('Segmentation Performance Over Time')
plt.ylim(0, 1)
plt.legend()

# Save the plot
plot_path = os.path.join(save_dir, 'avg_metrics_over_time_first_four_sessions.png')
plt.savefig(plot_path)
plt.close()

print(f'Plot saved to: {plot_path}')
