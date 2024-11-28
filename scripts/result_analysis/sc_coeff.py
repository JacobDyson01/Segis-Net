import os
import nibabel as nib
import numpy as np
import scipy.ndimage
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Directories
source_dir = '/home/groups/dlmrimnd/jacob/data/MND/source_images'  # Source images
target_dir = '/home/groups/dlmrimnd/jacob/data/MND/target_images'  # Target images (ground truth)
non_rdef_dir = '/home/groups/dlmrimnd/jacob/data/MND/non_rdef_fields'  # Non-rigid deformation fields output from model
save_dir = '/home/groups/dlmrimnd/jacob/data/graphs/metrics'

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Initialize lists to store results
spatial_correlation_scores = []
dice_scores = []

# Function to compute Spatial Correlation (SC)
def compute_spatial_correlation(ground_truth, predicted):
    # Flatten the 3D volumes to 1D arrays
    ground_truth_flat = ground_truth.flatten()
    predicted_flat = predicted.flatten()

    # Compute spatial correlation (Pearson correlation)
    correlation = np.corrcoef(ground_truth_flat, predicted_flat)[0, 1]
    return correlation

# Function to compute Dice coefficient
def compute_dice(ground_truth, predicted):
    intersection = np.sum(predicted[ground_truth == 1])
    dice = (2. * intersection) / (np.sum(ground_truth) + np.sum(predicted))
    return dice

# Function to apply the non_rdef deformation field to the source image
def apply_deformation(source_image, deformation_field):
    # Apply the deformation field using scipy's affine_transform or map_coordinates
    warped_image = scipy.ndimage.map_coordinates(source_image, deformation_field, order=1)
    return warped_image

# Iterate through the non_rdef directory to apply deformation fields
for subject_folder in os.listdir(non_rdef_dir):
    non_rdef_path = os.path.join(non_rdef_dir, subject_folder)
    source_path = os.path.join(source_dir, subject_folder)
    target_path = os.path.join(target_dir, subject_folder)

    if os.path.exists(non_rdef_path) and os.path.exists(source_path) and os.path.exists(target_path):
        # Load the source image
        source_file = os.path.join(source_path, 'source_image.nii.gz')
        source_img = nib.load(source_file)
        source_data = source_img.get_fdata()

        # Load the target image (ground truth)
        target_file = os.path.join(target_path, 'target_image.nii.gz')
        target_img = nib.load(target_file)
        target_data = target_img.get_fdata()

        # Load the non_rdef deformation field
        non_rdef_file = os.path.join(non_rdef_path, 'non_rdef.nii.gz')
        non_rdef_img = nib.load(non_rdef_file)
        non_rdef_data = non_rdef_img.get_fdata()

        # Apply the deformation field to the source image
        warped_source = apply_deformation(source_data, non_rdef_data)

        # Compute spatial correlation between warped source and target
        spatial_correlation = compute_spatial_correlation(target_data, warped_source)

        # Compute Dice coefficient between warped source and target segmentations (if binary masks)
        dice = compute_dice(target_data, warped_source)

        # Append results
        spatial_correlation_scores.append(spatial_correlation)
        dice_scores.append(dice)
    else:
        print(f'Data missing for {subject_folder}')

# Convert lists to DataFrame for visualization
data = pd.DataFrame({
    'Spatial Correlation': spatial_correlation_scores,
    'Dice': dice_scores  # Add Dice to the DataFrame
})

# Box plot for the distribution of Spatial Correlation and Dice across subjects
plt.figure(figsize=(8, 6))
sns.boxplot(data=data)
plt.ylim(0, 1)
plt.title('Distribution of Spatial Correlation and Dice Coefficient')
box_plot_path = os.path.join(save_dir, 'spatial_correlation_metrics.png')
plt.savefig(box_plot_path)
plt.close()

print(f'Registration accuracy (spatial correlation) box plot saved to: {box_plot_path}')
