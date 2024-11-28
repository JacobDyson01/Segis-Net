import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Lambda
from os.path import join
from tensorflow.keras.layers import Layer
from Transform_layer_interpn_0 import SpatialTransformer as Transformer
from SegNet_model_segGener import seg_net
from RegNet_model_regGener import reg_net

def joint_model(img_xyz, R_ch, S_ch, n_output, indexing='ij', alpha=0.2):
    # Define inputs
    tgt = Input(shape=img_xyz + (R_ch,), name='tgt_input')  # Target image
    src = Input(shape=img_xyz + (R_ch,), name='src_input')  # Source image
    S_src = Input(shape=img_xyz + (S_ch,), name='seg_input')  # Segmentation input
    aff_def = Input(shape=img_xyz + (3,), name='affine_input')  # Affine deformation field

    # Load segmentation and registration models
    seg_model = seg_net(img_xyz, S_ch, n_output, alpha=alpha)
    reg_model = reg_net(img_xyz, alpha=alpha)

    # Segmentation model applied to source segmentation input
    src_segm = seg_model(S_src)
    
    # Apply the registration model to the target, source, and affine deformation
    y, nonr_def = reg_model([tgt, src, aff_def])

    # Name the layers explicitly for use in the loss dictionary
    src_segm = Lambda(lambda x: x, name='srcSegm')(src_segm)
    y = Lambda(lambda x: x, name='warpedSrc')(y)
    nonr_def = Lambda(lambda x: x, name='nonr_def')(nonr_def)

    # Composite deformation of affine and non-linear
    all_def = Add(name='all_def')([nonr_def, aff_def])

    # Warp source segmentation
    tgt_segm = Transformer(interp_method='linear', indexing=indexing, name='movedSegm')([src_segm, all_def])

    # Define the model with named outputs
    model = Model(inputs=[tgt, src, S_src, aff_def], outputs=[y, tgt_segm, src_segm, nonr_def])

    return model


class DataGenerator(object):
    'Generates data for Keras'
    
    def __init__(self, dim_xyz, R_ch, S_ch, batch_size, n_output, shuffle=True):
        'Initialization'
        self.dim_xyz = dim_xyz  # Image dimensions (e.g., (160, 160, 160) for 3D images)
        self.R_ch = R_ch        # Number of channels in images to be registered (e.g., 1 for FA images)
        self.S_ch = S_ch        # Number of channels in the images to be segmented
        self.batch_size = batch_size
        self.n_output = n_output  # Number of output channels for segmentation (e.g., number of structures)
        self.shuffle = shuffle    # Optionally shuffle data between epochs
    
    def generate(self, part_index, R_path, S_path, segm_path, affine_path):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(part_index)
            imax = int(np.floor(len(indexes) / self.batch_size))  # Ensure we only process full batches
            
            for i in range(imax):
                # Ensure that the slice doesn't go out of bounds
                batch_indexes = indexes[i*self.batch_size:(i+1)*self.batch_size]

                # Safeguard against the batch being incomplete or empty
                if len(batch_indexes) == 0:
                    continue

                # Find list of IDs for the batch
                list_IDs_temp = [part_index[k] for k in batch_indexes]
                
                # Generate data
                x, y = self.__data_generation(list_IDs_temp, R_path, S_path, segm_path, affine_path)
                yield x, y

    def __get_exploration_order(self, part_index):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(part_index))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_IDs_temp, R_path, S_path, segm_path, affine_path):
        'Generates data of batch_size samples'
        # Initialization of arrays for registration and segmentation data
        R_tgt = np.zeros((self.batch_size, *self.dim_xyz, self.R_ch)).astype(dtype='float32')
        R_src = np.zeros((self.batch_size, *self.dim_xyz, self.R_ch)).astype(dtype='float32')
        S_src = np.zeros((self.batch_size, *self.dim_xyz, self.S_ch)).astype(dtype='float32')
        
        segm_tgt = np.zeros((self.batch_size, *self.dim_xyz, self.n_output)).astype(dtype='int8')
        segm_src = np.zeros((self.batch_size, *self.dim_xyz, self.n_output)).astype(dtype='int8')
        
        # 3 channels for the 3D affine deformation field
        aff_def = np.zeros((self.batch_size, *self.dim_xyz, 3)).astype(dtype='float32')
        zeros = np.zeros((self.batch_size, *self.dim_xyz, 3)).astype(dtype='float32')
        
        # Generate batch
        for i, ID in enumerate(list_IDs_temp):
            # Extract subject and session information
            subject_ses = ID.split('_')
            subject = subject_ses[0]
            ses1 = subject_ses[1]
            ses2 = subject_ses[2]

            # Load the target and source images
            tgt_p = join(R_path, f'{subject}_{ses1}_{ses2}', 'target_roi.nii.gz')
            src_p = join(R_path, f'{subject}_{ses1}_{ses2}', 'source_roi.nii.gz')
            tgt_img = nib.load(tgt_p).get_fdata().astype(dtype='float32')
            src_img = nib.load(src_p).get_fdata().astype(dtype='float32')

            # Load the source image to be segmented (the feature tensor) and the label (ground truth segmentation)
            tensor = src_img
            # Load the segmentation masks for the source and target
            segm1_p = join(segm_path, f'{subject}_{ses1}', 'binary_mask_roi.nii.gz')
            segm2_p = join(segm_path, f'{subject}_{ses2}', 'binary_mask_roi.nii.gz')
            segm1 = nib.load(segm1_p).get_fdata().astype(dtype='int8')
            segm2 = nib.load(segm2_p).get_fdata().astype(dtype='int8')

            # Normalize the tensor and images (zero mean, unit variance)
            tensor = (tensor - np.mean(tensor)) / np.std(tensor)
            tgt_img = (tgt_img - np.mean(tgt_img)) / np.std(tgt_img)
            src_img = (src_img - np.mean(src_img)) / np.std(src_img)

            # Expand dimensions for channels (necessary for input into the network)
            tensor = np.expand_dims(tensor, axis=-1)
            segm1 = np.expand_dims(segm1, axis=-1)
            segm2 = np.expand_dims(segm2, axis=-1)
            tgt_img = np.expand_dims(tgt_img, axis=-1)
            src_img = np.expand_dims(src_img, axis=-1)

            # Pre-estimated dense affine deformation field
            affine_p = join(affine_path, f'{subject}_{ses1}_{ses2}', 'deformation_field_roi.nii.gz')
            affine = nib.load(affine_p).get_fdata().astype(dtype='float32')
            affine = np.squeeze(affine)

            # Assign processed data to batch arrays
            S_src[i] = tensor
            R_tgt[i] = tgt_img
            R_src[i] = src_img
            segm_tgt[i] = segm1
            segm_src[i] = segm2
            aff_def[i] = affine

        # Return the batch of processed data
        return [R_tgt, R_src, S_src, aff_def], [R_tgt, segm_tgt, segm_src, zeros]
