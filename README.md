# Segis-Net Project

This repository contains the code and data for running the **Segis-Net** model, which performs image segmentation and registration, focusing on medical imaging tasks. Below is an overview of the different files and steps to get the model running, along with instructions for modifying various parameters like filenames, folder paths, and image sizes.

## Table of Contents
- [Overview](#overview)
- [File Structure](#file-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Modifying Parameters](#modifying-parameters)
  - [Image Size](#image-size)
  - [Filenames and Folder Paths](#filenames-and-folder-paths)

## Overview
Segis-Net is a deep learning-based model for 3D image segmentation and registration. It combines a segmentation model and a registration model to handle longitudinal MRI images and track changes in disease progression.

## File Structure

```bash
├── Segis-Net
│   ├── code/                            # Contains Segis-Net, Reg-Net, and Seg-Net model definitions
│   │   ├── Reg-Net/
│   │   ├── Seg-Net/
│   │   └── Segis-Net/                   # Main Segis-Net implementation
│   │       ├── Loss_metrics.py          # Custom loss functions (e.g., Dice Loss)
│   │       ├── prediction.py            # Script for generating predictions
│   │       ├── RegNet_model_regGener.py # Defines the registration network
│   │       ├── SegisNet_model_dataGenerator.py # Data generator for model training
│   │       ├── segment_MND_reversed.py  # Script for segmenting reversed MND data
│   │       ├── segment_MND.py           # Script for segmenting MND data
│   │       ├── segment_warped_output.py # Script for segmenting warped outputs
│   │       ├── SegNet_model_segGener.py # Defines the segmentation network
│   │       ├── tools.py                 # Utility functions
│   │       ├── train_index_new.npy      # Training indices for data split
│   │       ├── Train_SegisNet.py        # Main training script for Segis-Net
│   │       ├── Transform_layer_interpn_0.py # Spatial Transformer layer
│   │       └── vali_index_new.npy       # Validation indices for data split
│   ├── logs/                            # Logs generated during training
│   ├── scripts/                         # Preprocessing and utility scripts
│   │   ├── helpers/                     # Contains helper scripts and utilities
│   │   └── preprocessing/               # Preprocessing scripts (see detailed description below)
│   │       ├── ants_deformation_fields.sh  # Generates deformation fields with ANTs
│   │       ├── ants_warp_images.sh        # Warps input images using ANTs
│   │       ├── ants_warp_masks.sh         # Warps binary masks using ANTs
│   │       ├── crop_affine_roi.py         # Crops affine regions of interest
│   │       ├── crop_binary_masks_roi.py   # Crops binary masks for ROI
│   │       ├── crop_images_roi.py         # Crops images for ROI
│   │       ├── mask.py                    # Generates and processes masks
│   │       ├── rename.sh                  # Renames files to match naming conventions
│   │       ├── reversed_deformation_fields.sh # Reverses deformation fields using ANTs
│   │       ├── run_segmentation.sh        # Script to run segmentation
│   │       ├── train_segisNet.sh          # Wrapper script to train Segis-Net
│   │       └── training_split.py          # Splits data into training and validation sets
└── README.md                             # Documentation file

## Prerequisites

Ensure that the following are installed in your environment:

- Python 3.9
- TensorFlow (compatible with GPU, e.g., 2.4.0+)
- Keras (or TensorFlow Keras)
- ANTs (via Singularity image or installed on the system)
- CUDA 11.4 
- nibabel, numpy, scipy, and h5py libraries
- Any other dependencies listed in `environment.yml`

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your-username/Segis-Net.git
    cd Segis-Net
    ```

2. Set up a Conda environment and install dependencies:

    ```
    conda create --name segis-env python=3.9
    conda activate segis-env
    conda install tensorflow-gpu numpy nibabel h5py
    pip install -r requirements.txt
    ```

3. If using a GPU, ensure that CUDA and cuDNN are installed correctly and that TensorFlow is utilizing the GPU.

## How to Run

- **Train the Model:** To train the Segis-Net model, use the `train_SegisNet.py` script:

    ```
    python scripts/train_SegisNet.py
    ```

- **Generate Deformation Fields:** To generate deformation fields between warped source and target images using ANTs:

    ```
    sbatch scripts/generate_deformation_fields.sh
    ```

### Image Size

To modify the image size, you will need to change the `img_xyz` parameter passed in the model definition inside `joint_model.py`. This variable represents the 3D image dimensions (x, y, z).

img_xyz = (160, 160, 160) # Example of 3D image size

r


This can be adjusted in the following files:

- `joint_model.py`: In the `joint_model` function, update `img_xyz` with the desired shape.
- `train_SegisNet.py`: Ensure that any preprocessing matches the new image size.

### Filenames and Folder Paths

You can modify the folder paths and filenames to match your data structure. The main files to modify include:

- **`train_SegisNet.py`**: Modify the paths to your input data

### Pre-Processing Steps

Preprocessing Steps

The preprocessing pipeline prepares MRI data for use in Segis-Net by performing essential transformations such as creating binary masks, cropping, warping, and generating deformation fields. Follow the steps below to preprocess your data effectively:
Step 1: Create Binary Masks

The first step in preprocessing is to generate binary masks that isolate specific brain regions. This is done using the mask.py script, which processes MRI data in .mgz format to create .nii.gz binary masks.

    Input: MRI data files (e.g., aparc.DKTatlas+aseg.deep.withCC.mgz) located in subdirectories of the root folder.
    Output: Binary masks saved in the MND_binary_masks/ directory.

Customization:

You can modify the mask.py script to target different brain regions by adjusting the region values. For example:

    1024 and 2024 correspond to the precentral gyrus.
    2017 and 1017 correspond to the paracentral lobule.

Step 2: Rename Files

Standardize the naming of input files to match Segis-Net's expected input format. The rename.sh script ensures all files and directories follow a consistent naming convention.
Step 3: Crop Regions of Interest (ROI)

To reduce computation time and focus on relevant brain regions, crop both the input images and binary masks:

    Crop Input Images: Use crop_images_roi.py to crop regions of interest from the full-size input images.
    Crop Binary Masks: Use crop_binary_masks_roi.py to crop the binary masks to match the regions of interest.

Both cropped images and masks are saved in dedicated directories, such as warped_input_roi/ and warped_binary_masks/.
Step 4: Perform Image Registration and Warping

Align the input images and binary masks to a common template using ANTs tools:

    Warp Input Images: Warp the input images using transformation matrices generated by ANTs.
    Warp Binary Masks: Apply the same transformations to binary masks to maintain alignment.

Step 5: Generate Deformation Fields

Compute deformation fields between the source and target images to capture spatial transformations. These deformation fields are used for longitudinal analyses.
Step 6: (Optional) Reverse Deformation Fields

For longitudinal studies, you may need reversed deformation fields. This step can be performed using the corresponding script.
Step 7: Split Data for Training and Validation

Finally, divide the preprocessed data into training and validation sets using the training_split.py script. This ensures a balanced and reproducible dataset split.