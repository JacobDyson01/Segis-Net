import os
import numpy as np
import nibabel as nib
from keras.models import load_model

# Set the base directory for the data
data_dir = '/home/groups/dlmrimnd/jacob/data'

# Load the validation subjects
vali_subjects = np.load("/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/vali_index.npy")

# Path to the saved model (adjust path as necessary)
model_path = os.path.join(data_dir, 'results', 'accWeights.01-351.78.hdf5')

# Load the trained model
model = load_model(model_path)

# Function to load the subject's data (source and target images)
def load_subject_data(subject, data_dir):
    """
    Load source and target images for a subject.
    Modify paths as needed depending on your file structure.
    """
    subject_dir = os.path.join(data_dir, 'ants_warped_input_roi', subject)
    source_path = os.path.join(subject_dir, 'source_Warped.nii.gz')
    target_path = os.path.join(subject_dir, 'target_Warped.nii.gz')

    # Load the source and target images (assuming they are NIfTI files)
    source_img = nib.load(source_path).get_fdata()
    target_img = nib.load(target_path).get_fdata()

    return source_img, target_img

# Initialize lists to hold validation data
X_val = []
Y_val = []

# Load the validation data
print("Loading validation data...")
for subject in vali_subjects:
    source_img, target_img = load_subject_data(subject, data_dir)
    X_val.append(source_img)
    Y_val.append(target_img)

# Convert the lists to numpy arrays
X_val = np.array(X_val)
Y_val = np.array(Y_val)

# Ensure the data is in the correct shape (e.g., expand dimensions if necessary)
# This may depend on how your model expects input (e.g., add an extra dimension for channel)
X_val = np.expand_dims(X_val, axis=-1)  # Expand dimensions if your images are 3D
Y_val = np.expand_dims(Y_val, axis=-1)  # Do the same for target data if needed

# Evaluate the model on the validation data
print("Evaluating model on validation data...")
score = model.evaluate(X_val, Y_val, verbose=1)

# Print the validation loss and metrics
print(f"Validation Loss: {score[0]}")
print(f"Validation Metrics: {score[1:]}")  # Print all metrics after loss
