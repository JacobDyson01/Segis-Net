"""
Loss functions envolved in:
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021,
    paper: https://arxiv.org/abs/2012.14230
and our white matter tracts segmentation methods:
    Li et al., Neuro4Neuro: A neural network approach for neural tract segmentation 
    using large-scale population-based diffusion imaging. NeuroImage 2020.

please cite the paper if the code/method would be useful to your work.   

# for suggestions and questions, contact: BL (b.li@erasmusmc.nl)
"""
import keras.backend as K
from keras import losses
import tensorflow as tf
import numpy as np
from Transform_layer_interpn_0 import Grad


def sftDC(y_true, y_pred):
    epsilon=1e-07
    y_true_f = K.flatten(y_true)
    # softmax, two channel
    y_pred_f = K.reshape(y_pred, (-1, 2))
    y_pred_f = K.clip(y_pred_f, epsilon, 1. - epsilon)
    predict_binary = K.round(y_pred_f[:, 0])

    intersection = K.sum(y_true_f * predict_binary)
    union = K.sum(y_true_f) + K.sum(predict_binary)

    return K.mean(2. * intersection / union)

# weighted (3:1) inner product, following softmax
# the loss used in https://arxiv.org/abs/1908.10219
def wip(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.reshape(y_pred, (-1, 2))
    intersection = 3*K.mean(y_true_f * y_pred_f[:, 0]) + K.mean((1 - y_true_f) * y_pred_f[:, 1])

    return -intersection


def swip(y_true, y_pred):
    epsilon=1e-07
    
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)    

    intersection = 3*K.mean(y_true * y_pred) + K.mean((1 - y_true) * (1-y_pred))

    return -intersection


def sigmoid_sftDC(y_true, y_pred):
    epsilon=1e-07
    
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)    
    
    predict_binary = K.round(y_pred)

    intersection = 2 * K.sum(y_true * predict_binary)
    union = K.sum(y_true) + K.sum(predict_binary)

    return intersection / union


def sigmoid_DC(y_true, y_pred):
        
    predict_binary = K.round(y_pred)

    intersection = 2 * K.sum(y_true * predict_binary)
    union = K.sum(y_true) + K.sum(predict_binary)

    return intersection / union


def sigmoid_wip(y_true, y_pred):

    intersection = 3*K.mean(y_true * y_pred) + K.mean((1 - y_true) * (1 - y_pred))

    return -intersection


def sigmoid_swip(y_true, y_pred):
    epsilon=1e-07
    
    y_pred   = K.clip(y_pred, epsilon, 1. - epsilon)
    neg_pred = 1- y_pred
    neg_true = 1 - y_true

    intersection = 3*K.mean(y_true * y_pred) + K.mean(neg_true * neg_pred)

    return -intersection


def MSE_grad(y_true, y_pred):
    y = y_pred
    ndims = 3
    df = [None]*ndims
    for i in range(ndims):
        d = i+1
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = K.permute_dimensions(y, r)
        df[i] = y[1:, ...] - y[:-1, ...] 
    df = [tf.reduce_mean(f * f) for f in df]
    grad= tf.add_n(df) / len(df)
    
    mse = K.mean(K.square(y_true - y_pred))
    return mse+0.01*grad


def grad_loss(y_true, y_pred):
    return Grad('l2').loss(y_true, y_pred)


def clipMSE(y_true, y_pred, from_logits=False):
    epsilon=1e-07
    
    greater_1 = tf.where(tf.math.greater(y_pred, 1), y_pred, tf.zeros_like(y_pred))
    less_0    = tf.where(tf.math.less(y_pred, 0), y_pred, tf.zeros_like(y_pred))

    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)    

    MSE = losses.mean_squared_error(y_true, y_pred)
    penalty = K.sum(K.abs(less_0)) + K.sum(greater_1)
    
    return MSE+penalty


def MAE(y_true, y_pred):
    sigma = 0.035
    moved_img = y_pred
    tgt       = y_true
    return 1./sigma**2 *K.mean(K.abs(tgt - moved_img))


def MeanSquaredError(y_true, y_pred):
    
    return K.mean(K.square(y_true - y_pred))
    

def DC(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    predict_binary = K.round(y_pred_f)

    intersection = K.sum(y_true_f * predict_binary)
    union = K.sum(y_true_f) + K.sum(predict_binary)

    return K.mean(2. * intersection / union)

def DCLoss(y_true, y_pred):
    epsilon=1e-08
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)    

    intersection = - 2. * K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)

    return intersection / union
   

#https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
def focalBCE(y_true, y_pred, from_logits=False):
    gamma=1.
    alpha=1.25
    epsilon=1e-07

    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    
#    return -K.mean(alpha * y_true* K.pow(1. - y_pred, gamma) * K.log(y_pred))-K.mean((1-alpha) * (1.-y_true) *K.pow(y_pred, gamma) * K.log(1. - y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.mean(K.pow( pt_0, gamma) * K.log(1. - pt_0))


def clipBCE(y_true, y_pred, from_logits=False):
    epsilon=1e-07
    
    greater_1 = tf.where(tf.math.greater(y_pred, 1), y_pred, tf.zeros_like(y_pred))
    less_0    = tf.where(tf.math.less(y_pred, 0), y_pred, tf.zeros_like(y_pred))

    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)    

    BCE = losses.binary_crossentropy(y_true, y_pred)
    penalty = K.sum(K.abs(less_0)) + K.sum(greater_1)
    
    return BCE+penalty 


def cross_corr_multiScale(I,J):
    eps=1e-5
    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    win1 = [9] * ndims
    win2 = [5] * ndims
    win3 = [3] * ndims
    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I*I
    J2 = J*J
    IJ = I*J

    # compute filters
    #TODO fileter=(9,9,9,1,1), maybe should be (1,9,9,9,1)
    sum_filt1 = tf.ones([*win1, 1, 1])
    sum_filt2 = tf.ones([*win2, 1, 1])
    sum_filt3 = tf.ones([*win3, 1, 1])
    strides = [1] * (ndims + 2)
    padding = 'SAME'

    win_sizes = [np.prod(win1),np.prod(win2), np.prod(win3)]
    total_corr =0
    # compute local sums via convolution
    for i, sum_filt in enumerate([sum_filt1, sum_filt2, sum_filt3]):
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)
        
        win_size = win_sizes[i]
        numerator   = win_size*IJ_sum - I_sum*J_sum
        denom1_     = win_size*I2_sum - I_sum*I_sum
        denom2_     = win_size*J2_sum - J_sum*J_sum

        corr = tf.reduce_mean(numerator/tf.sqrt(denom1_*denom2_ + eps))
        total_corr += corr
    # return negative corr.
    return - total_corr/3 