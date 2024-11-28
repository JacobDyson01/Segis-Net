import tensorflow.keras.backend as K
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
from Transform_layer_interpn_0 import Grad

"""
This file contains various loss functions that are used for segmentation and registration tasks,
specifically in the context of deep learning models such as Segis-Net, which combines both tasks
for neuroimaging data, especially for white matter tracts segmentation in diffusion MRI.

Please cite the following papers if this code is useful to your work:
- Li et al., Longitudinal diffusion MRI analysis using Segis-Net. NeuroImage 2021.
- Li et al., Neuro4Neuro: A neural network approach for neural tract segmentation. NeuroImage 2020.
"""

def sftDC(y_true, y_pred):
    """
    Softmax-based Dice coefficient calculation for binary segmentation tasks.
    Dice Coefficient measures the overlap between the predicted and ground truth binary masks.
    """
    epsilon = 1e-07
    y_true_f = K.flatten(y_true)  # Flatten ground truth mask
    y_pred_f = K.reshape(y_pred, (-1, 2))  # Reshape predicted mask to two channels
    y_pred_f = K.clip(y_pred_f, epsilon, 1. - epsilon)  # Clip predictions to avoid numerical issues
    predict_binary = K.round(y_pred_f[:, 0])  # Binary prediction from softmax output

    intersection = K.sum(y_true_f * predict_binary)  # Calculate intersection
    union = K.sum(y_true_f) + K.sum(predict_binary)  # Calculate union

    return K.mean(2. * intersection / union)  # Dice coefficient

def wip(y_true, y_pred):
    """
    Weighted inner product loss function, used to encourage overlap between the predicted
    and true segmentations with a 3:1 weighting ratio. Used in segmentation tasks.
    """
    y_true_f = K.flatten(y_true)  # Flatten ground truth mask
    y_pred_f = K.reshape(y_pred, (-1, 2))  # Reshape predicted mask to two channels
    intersection = 3 * K.mean(y_true_f * y_pred_f[:, 0]) + K.mean((1 - y_true_f) * y_pred_f[:, 1])  # Weighted sum
    return -intersection  # Return negative inner product (for minimization)

def swip(y_true, y_pred):
    """
    Softmax-based weighted inner product loss with a 3:1 weighting. A variant of the
    wip loss function, but designed to work directly with softmax outputs.
    """
    epsilon = 1e-07
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions

    intersection = 3 * K.mean(y_true * y_pred) + K.mean((1 - y_true) * (1 - y_pred))  # Weighted sum
    return -intersection  # Return negative weighted inner product

def sigmoid_sftDC(y_true, y_pred):
    """
    Sigmoid-based Dice coefficient for binary segmentation tasks.
    """
    epsilon = 1e-07
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions
    
    predict_binary = K.round(y_pred)  # Apply binary threshold

    intersection = 2 * K.sum(y_true * predict_binary)  # Calculate intersection
    union = K.sum(y_true) + K.sum(predict_binary)  # Calculate union

    return intersection / union  # Return Dice coefficient

def sigmoid_DC(y_true, y_pred):
    """
    Sigmoid-based Dice coefficient for binary segmentation.
    """
    predict_binary = K.round(y_pred)  # Apply binary threshold
    intersection = 2 * K.sum(y_true * predict_binary)  # Calculate intersection
    union = K.sum(y_true) + K.sum(predict_binary)  # Calculate union

    return intersection / union  # Return Dice coefficient

def sigmoid_wip(y_true, y_pred):
    """
    Sigmoid-based weighted inner product for binary segmentation tasks.
    """
    intersection = 3 * K.mean(y_true * y_pred) + K.mean((1 - y_true) * (1 - y_pred))  # Weighted sum
    return -intersection  # Return negative weighted inner product

def sigmoid_swip(y_true, y_pred):
    """
    Sigmoid-based weighted inner product with clipping for binary segmentation.
    """
    epsilon = 1e-07
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions
    neg_pred = 1 - y_pred  # Calculate negative prediction
    neg_true = 1 - y_true  # Calculate negative ground truth

    intersection = 3 * K.mean(y_true * y_pred) + K.mean(neg_true * neg_pred)  # Weighted sum
    return -intersection  # Return negative weighted inner product

def MSE_grad(y_true, y_pred):
    """
    Mean Squared Error loss with gradient regularization for smoothness.
    """
    y = y_pred
    ndims = 3
    df = [None] * ndims
    for i in range(ndims):
        d = i + 1
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = K.permute_dimensions(y, r)  # Permute dimensions for gradient calculation
        df[i] = y[1:, ...] - y[:-1, ...]  # Calculate differences between adjacent pixels
    df = [tf.reduce_mean(f * f) for f in df]  # Sum of squared differences
    grad = tf.add_n(df) / len(df)  # Average gradient

    mse = K.mean(K.square(y_true - y_pred))  # Mean squared error
    return mse + 0.01 * grad  # Add gradient regularization

def grad_loss(y_true, y_pred):
    """
    Gradient loss for smoothness using custom transformation layer.
    """
    return Grad('l2').loss(y_true, y_pred)  # Apply l2 gradient loss from custom layer

def clipMSE(y_true, y_pred, from_logits=False):
    """
    Mean Squared Error with clipping for values outside of [0, 1] range.
    """
    epsilon = 1e-07
    greater_1 = tf.where(tf.math.greater(y_pred, 1), y_pred, tf.zeros_like(y_pred))
    less_0 = tf.where(tf.math.less(y_pred, 0), y_pred, tf.zeros_like(y_pred))

    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions to [epsilon, 1 - epsilon]
    
    MSE = losses.mean_squared_error(y_true, y_pred)  # Mean squared error
    penalty = K.sum(K.abs(less_0)) + K.sum(greater_1)  # Penalty for out-of-bounds values

    return MSE + penalty  # Return MSE with penalty

def MAE(y_true, y_pred):
    """
    Mean Absolute Error loss with sigma scaling.
    """
    sigma = 0.035
    return 1. / sigma ** 2 * K.mean(K.abs(y_true - y_pred))  # Return scaled MAE

def MeanSquaredError(y_true, y_pred):
    """
    Simple Mean Squared Error calculation.
    """
    return K.mean(K.square(y_true - y_pred))  # Return mean squared error

def DC(y_true, y_pred):
    """
    Basic Dice Coefficient calculation for binary segmentation.
    """
    y_true_f = K.flatten(y_true)  # Flatten ground truth
    y_pred_f = K.flatten(y_pred)  # Flatten predictions
    predict_binary = K.round(y_pred_f)  # Binary prediction

    intersection = K.sum(y_true_f * predict_binary)  # Intersection
    union = K.sum(y_true_f) + K.sum(predict_binary)  # Union

    return K.mean(2. * intersection / union)  # Dice coefficient

def DCLoss(y_true, y_pred):
    # Cast y_true to float32
    y_true = K.cast(y_true, dtype='float32')
    
    # Compute Dice Coefficient Loss
    intersection = 2. * K.sum(y_true * y_pred)  # Intersection
    sum_val = K.sum(y_true) + K.sum(y_pred)  # Sum of y_true and y_pred
    smooth = 1e-6  # Small constant to avoid division by zero
    
    return 1 - (intersection + smooth) / (sum_val + smooth)

def focalBCE(y_true, y_pred, from_logits=False):
    """
    Focal Binary Cross Entropy (BCE) Loss to handle class imbalance in segmentation.
    """
    gamma = 1.
    alpha = 1.25
    epsilon = 1e-07
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))  # Positive class
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))  # Negative class

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(K.pow(pt_0, gamma) * K.log(1. - pt_0))  # Focal loss

def clipBCE(y_true, y_pred, from_logits=False):
    """
    Binary Cross Entropy with clipping for predictions outside [0, 1].
    """
    epsilon = 1e-07
    greater_1 = tf.where(tf.math.greater(y_pred, 1), y_pred, tf.zeros_like(y_pred))
    less_0 = tf.where(tf.math.less(y_pred, 0), y_pred, tf.zeros_like(y_pred))

    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions

    BCE = losses.binary_crossentropy(y_true, y_pred)  # Binary cross entropy
    penalty = K.sum(K.abs(less_0)) + K.sum(greater_1)  # Penalty for out-of-bounds predictions

    return BCE + penalty  # Return BCE with penalty

def cross_corr_multiScale(I, J):
    """
    Multi-scale cross-correlation for comparing two image volumes, I and J.
    """
    eps = 1e-5
    ndims = len(I.get_shape().as_list()) - 2  # Number of spatial dimensions
    assert ndims in [1, 2, 3], "Volumes should be 1 to 3 dimensions. Found: %d" % ndims
    win1 = [9] * ndims
    win2 = [5] * ndims
    win3 = [3] * ndims
    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)  # Get convolution function based on dimensions

    # Compute local sums
    I2 = I * I
    J2 = J * J
    IJ = I * J

    # Define filters for multi-scale correlation
    sum_filt1 = tf.ones([*win1, 1, 1])
    sum_filt2 = tf.ones([*win2, 1, 1])
    sum_filt3 = tf.ones([*win3, 1, 1])
    strides = [1] * (ndims + 2)
    padding = 'SAME'

    win_sizes = [np.prod(win1), np.prod(win2), np.prod(win3)]
    total_corr = 0

    # Compute multi-scale cross-correlation
    for i, sum_filt in enumerate([sum_filt1, sum_filt2, sum_filt3]):
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)
        
        win_size = win_sizes[i]
        numerator = win_size * IJ_sum - I_sum * J_sum
        denom1_ = win_size * I2_sum - I_sum * I_sum
        denom2_ = win_size * J2_sum - J_sum * J_sum

        corr = tf.reduce_mean(numerator / tf.sqrt(denom1_ * denom2_ + eps))
        total_corr += corr

    return - total_corr / 3  # Return average negative correlation
