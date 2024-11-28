"""
# Seg-Net model for concurrent (cross-sectional) segmentation of multiple 
structures, developed in: 
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021.
paper: https://arxiv.org/abs/2012.14230

please cite the paper if the code/method would be useful to your work.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, LeakyReLU
import numpy as np
from os.path import join


def ConvBlockA(x, ch_1, ch_2, alpha):
    """
    Convolutional block A with two 3D convolution layers followed by Batch Normalization and LeakyReLU activation.
    """
    out = x    
    for i in [ch_1, ch_2]:
        out = Conv3D(i, (3, 3, 3), use_bias=False, padding='same')(out)  # 3x3x3 Conv layer
        out = BatchNormalization(epsilon=0.001, momentum=0.9)(out)
        out = LeakyReLU(alpha=alpha)(out)               

    return out


def ConvBlockC(x, ch_1, n_output, alpha):
    """
    Convolutional block C with multiple convolution layers and sub-branching for output layers.
    """
    ch_1 = int(ch_1)
    n_output = int(n_output)
    ch_2 = int((ch_1 + n_output - 1) // n_output)
    
    out = Conv3D(ch_1, (3, 3, 3), use_bias=False, padding='same')(x)
    out = BatchNormalization(epsilon=0.001, momentum=0.9)(out)
    out = LeakyReLU(alpha=alpha)(out)

    output = []
    for i in range(n_output):
        out_i = Conv3D(ch_2, (3, 3, 3), use_bias=False, padding='same')(out)
        out_i = BatchNormalization(epsilon=0.001, momentum=0.9)(out_i)
        out_i = LeakyReLU(alpha=alpha)(out_i)
        out_i = Conv3D(1, (1, 1, 1), activation="sigmoid", padding='same')(out_i)
        output.append(out_i)
    
    return output


def seg_net(img_xyz, img_ch, n_output, alpha=0.2):
    """
    Seg-Net model for multi-structure segmentation of 3D images.
    """
    num_start_ch = 16

    """ Input layer """
    tensors = Input(shape=(*img_xyz, img_ch), name='tensor_input')

    """ Encoder """
    conv_1 = ConvBlockA(tensors, num_start_ch, num_start_ch, alpha)
    pool_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_1)

    conv_2 = ConvBlockA(pool_1, num_start_ch * 2, num_start_ch * 2, alpha)
    pool_2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_2)

    conv_3 = ConvBlockA(pool_2, num_start_ch * 4, num_start_ch * 4, alpha)
    pool_3 = MaxPooling3D(pool_size=(2, 2, 2))(conv_3)
    
    conv_4 = ConvBlockA(pool_3, num_start_ch * 8, num_start_ch * 8, alpha)
    pool_4 = MaxPooling3D(pool_size=(2, 2, 2))(conv_4)

    """ Bottleneck Layer """
    conv_5 = ConvBlockA(pool_4, num_start_ch * 16, num_start_ch * 16, alpha)
    up_6 = UpSampling3D(size=(2, 2, 2))(conv_5)

    """ Decoder """
    up_6 = concatenate([up_6, conv_4], axis=4)
    conv_6 = ConvBlockA(up_6, num_start_ch * 8, num_start_ch * 8, alpha)
    up_7 = UpSampling3D(size=(2, 2, 2))(conv_6)

    up_7 = concatenate([up_7, conv_3], axis=4)
    conv_7 = ConvBlockA(up_7, num_start_ch * 4, num_start_ch * 4, alpha)
    up_8 = UpSampling3D(size=(2, 2, 2))(conv_7)

    up_8 = concatenate([up_8, conv_2], axis=4)
    conv_8 = ConvBlockA(up_8, num_start_ch * 2, num_start_ch * 2, alpha)
    up_9 = UpSampling3D(size=(2, 2, 2))(conv_8)

    up_9 = concatenate([up_9, conv_1], axis=4)

    """ Sub-branch output """
    out = ConvBlockC(up_9, num_start_ch, n_output, alpha)

    # Define the Seg-Net model
    model = Model(inputs=tensors, outputs=out)
    
    return model


class DataGenerator(object):
    'Generates data for TensorFlow/Keras models'

    def __init__(self, dim_xyz, dim_ch, batch_size, n_output, shuffle=True):
        self.dim_xyz = dim_xyz  # Image dimensions
        self.dim_ch = dim_ch    # Number of feature channels in the images to be segmented
        self.batch_size = batch_size
        self.n_output = n_output  # Number of output segmentation channels
        self.shuffle = shuffle  # Shuffle data between epochs

    def generate(self, part_index, img_path, segm_path):
        'Generates batches of samples'
        while 1:
            indexes = self.__get_exploration_order(part_index)
            imax = int(len(indexes) / self.batch_size)

            for i in range(imax):
                list_IDs_temp = [part_index[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                x, y = self.__data_generation(list_IDs_temp, img_path, segm_path)
                yield x, y

    def __get_exploration_order(self, part_index):
        'Generates order of exploration'
        indexes = np.arange(len(part_index))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_IDs_temp, img_path, segm_path):
        'Generates data of batch_size samples'
        img_src = np.zeros((self.batch_size, *self.dim_xyz, self.dim_ch)).astype('float32')
        segm_src = np.zeros((self.batch_size, *self.dim_xyz, self.n_output)).astype('int8')

        for i, ID in enumerate(list_IDs_temp):
            img_p = join(img_path, str(ID), 'src_FA.nii.gz')
            img = nib.load(img_p).get_fdata().astype('float32')

            # Load the ground truth image
            segm_p = join(segm_path, str(ID), 'gt_FA.nii.gz')
            segm = nib.load(segm_p).get_fdata().astype('float32')

            # Normalize the image
            img -= np.mean(img)
            img /= np.std(img)
            img_src[i, :, :, :, :] = img

            segm_src[i, :, :, :, :] = segm

        return img_src, segm_src
