import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import LeakyReLU
from Transform_layer_interpn_0 import SpatialTransformer

def ConvBlockA(x, ch_1, ch_2, alpha):
    """
    Convolutional block A: Applies two 3D convolution layers with Batch Normalization and LeakyReLU activation.
    """
    out = x    
    for i in [ch_1, ch_2]:
        out = Conv3D(i, (3, 3, 3), use_bias=False, padding='same', 
                     kernel_initializer='he_normal')(out)  # use he_normal initializer
        out = BatchNormalization(epsilon=0.001, momentum=0.9)(out)
        out = LeakyReLU(alpha=alpha)(out)
    return out

def ConvBlockB(x, ch_1, ch_2, ch_3, alpha):
    """
    Convolutional block B: Applies three 3D convolution layers with Batch Normalization and LeakyReLU activation.
    """
    out = x    
    for i in [ch_1, ch_2, ch_3]:
        out = Conv3D(i, (3, 3, 3), use_bias=False, padding='same', 
                     kernel_initializer='he_normal')(out)  # use he_normal initializer
        out = BatchNormalization(epsilon=0.001, momentum=0.9)(out)
        out = LeakyReLU(alpha=alpha)(out)
    return out

def reg_net(img_xyz, alpha=0.2):
    """
    Registration network for 3D image registration, including both affine and non-linear deformation fields.
    """
    num_start_ch = 16
    
    """ Input Layers """
    tgt = Input(shape=(*img_xyz, 1))  # Target image (1 channel)
    src = Input(shape=(*img_xyz, 1))  # Source image (1 channel)
    aff_def = Input(shape=(*img_xyz, 3))  # Affine deformation (3 channels for x, y, z displacements)
    
    """ Affine Warping """
    # Apply affine deformation to the source image using SpatialTransformer
    aff_warped = SpatialTransformer(interp_method='linear', indexing='ij', 
                                    name='affine_warped')([src, aff_def])
    
    """ Non-linear Deformation Field """
    # After affine warping, apply non-linear deformation
    # Pass through the network
    inputs = concatenate([tgt, aff_warped], axis=-1)
    
    """ Encoder (Downsampling Path) """
    conv_1 = ConvBlockA(inputs, int(num_start_ch / 2), num_start_ch, alpha)
    pool_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_1)

    conv_2 = ConvBlockA(pool_1, num_start_ch * 2, num_start_ch * 2, alpha)
    pool_2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_2)
    
    conv_3 = ConvBlockA(pool_2, num_start_ch * 4, num_start_ch * 4, alpha)
    pool_3 = MaxPooling3D(pool_size=(2, 2, 2))(conv_3)
    
    conv_4 = ConvBlockA(pool_3, num_start_ch * 8, num_start_ch * 8, alpha)
    pool_4 = MaxPooling3D(pool_size=(2, 2, 2))(conv_4)

    """ Bottleneck Layer """
    conv_5 = ConvBlockB(pool_4, num_start_ch * 16, num_start_ch * 8, num_start_ch * 8, alpha)
    up_6 = UpSampling3D(size=(2, 2, 2))(conv_5)

    """ Decoder (Upsampling Path) """
    up_6 = concatenate([up_6, conv_4], axis=4)
    conv_6 = ConvBlockB(up_6, num_start_ch * 8, num_start_ch * 4, num_start_ch * 4, alpha)
    up_7 = UpSampling3D(size=(2, 2, 2))(conv_6)
    
    up_7 = concatenate([up_7, conv_3], axis=4)
    conv_7 = ConvBlockB(up_7, num_start_ch * 4, num_start_ch * 2, num_start_ch * 2, alpha)
    up_8 = UpSampling3D(size=(2, 2, 2))(conv_7)
    
    up_8 = concatenate([up_8, conv_2], axis=4)
    conv_8 = ConvBlockB(up_8, num_start_ch * 2, num_start_ch, num_start_ch, alpha)
    up_9 = UpSampling3D(size=(2, 2, 2))(conv_8)
    
    up_9 = concatenate([up_9, conv_1], axis=4)
    conv_9 = ConvBlockA(up_9, num_start_ch, int(num_start_ch / 2), alpha)
    
    """ Non-linear Deformation Field Prediction """
    nonr_def = Conv3D(3, (3, 3, 3), activation=None, padding='same', 
                      kernel_initializer='he_normal',  # use he_normal initializer
                      name='nonr_def')(conv_9)
    
    """ Apply the Non-linear Deformation """
    # Apply the non-linear deformation to the affine-warped image
    final_warped = SpatialTransformer(interp_method='linear', indexing='ij', 
                                      name='final_warped')([aff_warped, nonr_def])
    
    """ Output Model """
    model = Model(inputs=[tgt, src, aff_def], outputs=[final_warped, nonr_def])
    
    return model
