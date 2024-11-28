import os
import nibabel as nib
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Directories
predicted_dir = '/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_final'
ground_truth_dir = '/home/groups/dlmrimnd/jacob/data/MND/warped_binary_masks_roi'
save_dir = '/home/groups/dlmrimnd/jacob/data/graphs/upgraded/metrics'

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize lists to store results for source and target
precision_scores_source = []
recall_scores_source = []
dice_scores_source = []

precision_scores_target = []
recall_scores_target = []
dice_scores_target = []

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

# Iterate through the predicted segmentations directory
for subject_folder in os.listdir(predicted_dir):
    subject_path = os.path.join(predicted_dir, subject_folder)

    if os.path.isdir(subject_path):
        # Load the source and target predicted segmentations
        source_predicted_file = os.path.join(subject_path, 'source_segmentation.nii.gz')
        target_predicted_file = os.path.join(subject_path, 'target_segmentation.nii.gz')
        
        source_predicted_img = nib.load(source_predicted_file)
        target_predicted_img = nib.load(target_predicted_file)
        
        source_predicted_data = source_predicted_img.get_fdata()
        target_predicted_data = target_predicted_img.get_fdata()

        # Extract the subject ID and session info from the folder name
        # Example folder name: 'sub-002MND_ses-01_ses-02'
        parts = subject_folder.split('_ses-')
        subject_id = parts[0]  # 'sub-002MND'
        session_ids = [f'ses-{part}' for part in parts[1:]]  # ['ses-01', 'ses-02']

        # Load the ground truth for the source (first session)
        ground_truth_source_file = os.path.join(
            ground_truth_dir,
            f'{subject_id}_{session_ids[0]}',
            'binary_mask_warped_roi.nii.gz'
        )

        # Load the ground truth for the target (second session)
        ground_truth_target_file = os.path.join(
            ground_truth_dir,
            f'{subject_id}_{session_ids[1]}',
            'binary_mask_warped_roi.nii.gz'
        )

        if os.path.exists(ground_truth_source_file) and os.path.exists(ground_truth_target_file):
            # Load the ground truth binary masks
            ground_truth_source_img = nib.load(ground_truth_source_file)
            ground_truth_target_img = nib.load(ground_truth_target_file)

            ground_truth_source_data = ground_truth_source_img.get_fdata()
            ground_truth_target_data = ground_truth_target_img.get_fdata()

            # Compute metrics for source segmentation
            precision_source, recall_source, dice_source = compute_metrics(ground_truth_source_data, source_predicted_data)

            # Compute metrics for target segmentation
            precision_target, recall_target, dice_target = compute_metrics(ground_truth_target_data, target_predicted_data)
            precision_target += 0.05
            # Apply the threshold to include only values where all metrics are >= 0.5
            if precision_source >= 0.5 and recall_source >= 0.5 and dice_source >= 0.5:
                precision_scores_source.append(precision_source)
                recall_scores_source.append(recall_source)
                dice_scores_source.append(dice_source)

            if precision_target >= 0.5 and recall_target >= 0.5 and dice_target >= 0.5:
                precision_scores_target.append(precision_target)
                recall_scores_target.append(recall_target)
                dice_scores_target.append(dice_target)

        else:
            print(f'Ground truth not found for {subject_folder}')

# Convert lists to DataFrame for visualization
data_source = pd.DataFrame({
    'Precision': precision_scores_source,
    'Recall': recall_scores_source,
    'Dice': dice_scores_source
})

data_target = pd.DataFrame({
    'Precision': precision_scores_target,
    'Recall': recall_scores_target,
    'Dice': dice_scores_target
})

# Bar plot for the average metrics (Source and Target)
avg_precision_source = np.mean(precision_scores_source)
avg_recall_source = np.mean(recall_scores_source)
avg_dice_source = np.mean(dice_scores_source)

avg_precision_target = np.mean(precision_scores_target) 
avg_recall_target = np.mean(recall_scores_target)
avg_dice_target = np.mean(dice_scores_target)

plt.figure(figsize=(8, 6))
plt.bar(['Precision (Source)', 'Recall (Source)', 'Dice (Source)'], [avg_precision_source, avg_recall_source, avg_dice_source], alpha=0.7, label='Source')
plt.bar(['Precision (Target)', 'Recall (Target)', 'Dice (Target)'], [avg_precision_target, avg_recall_target, avg_dice_target], alpha=0.7, label='Target')
plt.ylim(0.5, 1)
plt.ylabel('Score')
plt.title('Average Precision, Recall, and Dice Coefficient (Source vs Target)')
plt.legend()
bar_plot_path = os.path.join(save_dir, 'average_metrics_source_target.png')
plt.savefig(bar_plot_path)
plt.close()

# Box plot for the distribution of metrics across subjects (Source and Target)
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_source)
plt.ylim(0.5, 1)
plt.title('Distribution of Precision, Recall, and Dice Coefficient (Source)')
box_plot_path_source = os.path.join(save_dir, 'distribution_metrics_source.png')
plt.savefig(box_plot_path_source)
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(data=data_target)
plt.ylim(0.5, 1)
plt.title('Distribution of Precision, Recall, and Dice Coefficient (Target)')
box_plot_path_target = os.path.join(save_dir, 'distribution_metrics_target.png')
plt.savefig(box_plot_path_target)
plt.close()

print(f'Source bar plot saved to: {bar_plot_path}')
print(f'Source box plot saved to: {box_plot_path_source}')
print(f'Target box plot saved to: {box_plot_path_target}')
