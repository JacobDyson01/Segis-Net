import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import nibabel as nib
import matplotlib.pyplot as plt

# Directory where BElong dataset images are stored
mnd_directory = '/path/to/mnd_directory'

# Path to the saved model weights
model_weights_path = '/home/groups/dlmrimnd/jacob/data/saved_results/Run_1_OOM/model_weight_out.h5'

# Load the model architecture and compile
# (Replace `your_model_architecture()` with your actual model architecture function)
def your_model_architecture(input_shape):
    # Define the model architecture here
    # For example, using Segis-Net architecture:
    model = joint_model(input_shape, R_ch=1, S_ch=1, n_output=1, indexing='ij', alpha=0.2)
    return model

# Define the image dimensions (e.g., based on the model input requirements)
input_shape = (144, 96, 112)  # Example, modify based on your input image size

# Initialize the model
model = your_model_architecture(input_shape)

# Load the saved weights into the model
model.load_weights(model_weights_path)
print("Model weights loaded successfully.")

# Function to load a .nii.gz image and preprocess it
def load_and_preprocess_image(image_path):
    image = nib.load(image_path)
    image_data = image.get_fdata()
    # Normalize or preprocess the image as needed
    # For example, you can normalize to [0, 1] range:
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    # Resize or crop if necessary to match the input shape
    # Assuming you need (144, 96, 112), reshape if required:
    if image_data.shape != input_shape:
        image_data = np.resize(image_data, input_shape)
    return image_data

# Function to segment an image using the model
def segment_image(image):
    # Expand dimensions to fit the model's input shape (batch_size, H, W, D, channels)
    image_expanded = np.expand_dims(image, axis=0)
    segmented_image = model.predict(image_expanded)
    # Post-process the output if necessary (e.g., thresholding)
    segmentation_mask = (segmented_image[0] > 0.5).astype(np.uint8)  # Threshold at 0.5
    return segmentation_mask

# Loop over images in the BElong dataset and segment each one
for filename in os.listdir(mnd_directory):
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(mnd_directory, filename)
        print(f"Processing {filename}...")
        
        # Load and preprocess the image
        image = load_and_preprocess_image(image_path)
        
        # Segment the image
        segmentation_mask = segment_image(image)
        
        # Save the segmentation mask as a .nii.gz file
        output_path = os.path.join(mnd_directory, f'segmented_{filename}')
        segmentation_img = nib.Nifti1Image(segmentation_mask, affine=np.eye(4))  # Modify affine if necessary
        nib.save(segmentation_img, output_path)
        print(f"Segmentation saved for {filename} at {output_path}")
        
print("Segmentation of BElong dataset completed.")
