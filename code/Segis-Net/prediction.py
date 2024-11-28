import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to load the NIfTI image
def load_nifti_image(file_path):
    """
    Load a NIfTI file and return its image data, affine, and header.
    """
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data, img.affine, img.header

# Function to save NIfTI image
def save_nifti_image(output_path, image_data, affine, header):
    """
    Save an image as a NIfTI file with the given affine transformation and header.
    """
    new_img = nib.Nifti1Image(image_data, affine, header)
    nib.save(new_img, output_path)

# Function to preprocess the image before passing to the model
def preprocess_image(image_data):
    """
    Preprocess the image by normalizing and reshaping to the required input dimensions of the model.
    """
    image_data = image_data / np.max(image_data)  # Normalize image
    image_data = np.expand_dims(image_data, axis=-1)  # Add a channel dimension (if necessary)
    image_data = np.expand_dims(image_data, axis=0)   # Add batch dimension
    return image_data

# Function to postprocess the segmentation output
def postprocess_output(output):
    """
    Postprocess the segmentation output by removing extra dimensions and binarizing the output.
    """
    output = np.squeeze(output, axis=0)   # Remove batch dimension
    output = np.squeeze(output, axis=-1)  # Remove channel dimension
    output = (output > 0.5).astype(np.uint8)  # Binarize the output with a threshold of 0.5
    return output

# Main segmentation function
def segment_and_save(model_path, input_image_path, output_segmentation_path):
    """
    Load a model and perform segmentation on a NIfTI image, saving the output.
    """
    # Load the trained model using TensorFlow 2.x
    model = load_model(model_path)

    # Load and preprocess the input image
    image_data, affine, header = load_nifti_image(input_image_path)
    preprocessed_image = preprocess_image(image_data)

    # Perform segmentation using the model
    predicted_output = model.predict(preprocessed_image)

    # Postprocess the output to obtain the segmentation
    segmented_output = postprocess_output(predicted_output)

    # Save the segmentation output as a NIfTI file
    save_nifti_image(output_segmentation_path, segmented_output, affine, header)

    print(f"Segmentation saved to: {output_segmentation_path}")

# Example usage
model_path = "/home/groups/dlmrimnd/jacob/data/combined_data/saved_results/run_proper_3/model_weight_out.h5"
input_image_path = "/home/groups/dlmrimnd/jacob/data/combined_data/warped_input_roi/sub-002MND_ses-01_ses-02/source_Warped_roi.nii.gz"
output_segmentation_path = "/home/groups/dlmrimnd/jacob/data/test_segmentation.nii.gz"

segment_and_save(model_path, input_image_path, output_segmentation_path)
