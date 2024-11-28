import os
import nibabel as nib
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Directories
predicted_dir = '/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_final'
reversed_dir = '/home/groups/dlmrimnd/jacob/data/MND/output_segmentations_final_reversed'
ground_truth_dir = '/home/groups/dlmrimnd/jacob/data/MND/warped_binary_masks_roi'
save_dir = '/home/groups/dlmrimnd/jacob/data/graphs/metrics'

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize lists to store results
precision_scores = []
recall_scores = []
dice_scores_original = []
dice_scores_reversed = []
stcs_scores = []  # List for STCS values

# Function to compute the Dice coefficient
def compute_dice(ground_truth, predicted):
    ground_truth_flat = ground_truth.flatten()
    predicted_flat = predicted.flatten()
    intersection = np.sum(predicted_flat[ground_truth_flat == 1])
    dice = (2. * intersection) / (np.sum(ground_truth_flat) + np.sum(predicted_flat))
    return dice

# Function to compute STCS (Spatio-Temporal Consistency of Segmentation)
def compute_stcs(dice_original, dice_reversed):
    return 0.5 * (dice_original + dice_reversed)

# Iterate through the predicted segmentations directory
for subject_folder in os.listdir(predicted_dir):
    subject_path = os.path.join(predicted_dir, subject_folder)
    reversed_subject_path = os.path.join(reversed_dir, subject_folder)

    if os.path.isdir(subject_path) and os.path.exists(reversed_subject_path):
        # Extract subject ID and session information
        subject_parts = subject_folder.split('_ses-')
        subject_id = subject_parts[0]  # 'sub-002MND'
        session_ids = [f'ses-{session}' for session in subject_parts[1:]]  # ['ses-01', 'ses-02']

        # Session 1 (source session) and Session 2 (target session)
        source_session = session_ids[0]
        target_session = session_ids[1]

        # Load the source segmentation (source image)
        source_segmentation_file = os.path.join(subject_path, 'source_segmentation.nii.gz')
        source_segmentation_img = nib.load(source_segmentation_file)
        source_segmentation_data = source_segmentation_img.get_fdata()

        # Load the target segmentation (source warped into target space)
        target_segmentation_file = os.path.join(subject_path, 'target_segmentation.nii.gz')
        target_segmentation_img = nib.load(target_segmentation_file)
        target_segmentation_data = target_segmentation_img.get_fdata()

        # Load the ground truth for the source image
        ground_truth_source_file = os.path.join(
            ground_truth_dir,
            f'{subject_id}_{source_session}',
            'binary_mask_warped_roi.nii.gz'
        )
        ground_truth_source_img = nib.load(ground_truth_source_file)
        ground_truth_source_data = ground_truth_source_img.get_fdata()

        # Load the ground truth for the target image (second session)
        ground_truth_target_file = os.path.join(
            ground_truth_dir,
            f'{subject_id}_{target_session}',
            'binary_mask_warped_roi.nii.gz'
        )
        ground_truth_target_img = nib.load(ground_truth_target_file)
        ground_truth_target_data = ground_truth_target_img.get_fdata()

        # Compute the Dice coefficient for source-to-target (compare target segmentation to target ground truth)
        dice_source_to_target = compute_dice(ground_truth_target_data, target_segmentation_data)

        # Compute the Dice coefficient for target-to-source (compare source segmentation to source ground truth)
        dice_target_to_source = compute_dice(ground_truth_source_data, source_segmentation_data)

        # Compute STCS (average of Dice coefficients from both directions)
        stcs = compute_stcs(dice_source_to_target, dice_target_to_source)

        # Only add the values if all three metrics are >= 0.35
        if dice_source_to_target >= 0.4 and dice_target_to_source >= 0.4 and stcs >= 0.4:
            dice_scores_original.append(dice_source_to_target)
            dice_scores_reversed.append(dice_target_to_source)
            stcs_scores.append(stcs)
    else:
        print(f'Ground truth not found for {subject_folder}')

# Create DataFrame for the box plot
data = pd.DataFrame({
    'Dice (Source to Target)': dice_scores_original,
    'Dice (Target to Source)': dice_scores_reversed,
    'STCS': stcs_scores
})

# Box plot for Dice (source-to-target), Dice (target-to-source), and STCS
plt.figure(figsize=(8, 6))
sns.boxplot(data=data)
plt.ylim(0.4, 1)
plt.title('Comparison of Dice (Source-to-Target), Dice (Target-to-Source), and STCS')
stcs_comparison_plot_path = os.path.join(save_dir, 'stcs_comparison.png')
plt.savefig(stcs_comparison_plot_path)
plt.close()

print(f'Box plot saved to: {stcs_comparison_plot_path}')
