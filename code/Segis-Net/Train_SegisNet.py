"""
# tensorflow/keras training for Segis-Net
for non-linear registration and cocurrent segmentation of multiple white matter tracts
devoloped in :
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021.
paper: https://arxiv.org/abs/2012.14230

please cite the paper if the code/method would be useful to your work.

# for suggestions and questions, contact: BL (b.li@erasmusmc.nl)
"""

import os
# the number of gpus to use in parallel; 
G = 1
if 1==G:
    # specify to use one GPU only, e.g., either '0' or '1'
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
from os.path import join, exists
import numpy as np
#import csv
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

#import time
from keras.optimizers import Adam
import keras.backend as K
from tools import Adaptive_LossWeight, LossHistory_basic
from Loss_metrics import sigmoid_DC, sigmoid_sftDC, grad_loss, DCLoss, MeanSquaredError
from SegisNet_model_dataGenerator import joint_model, DataGenerator
import matplotlib.pyplot as plt

""" Initial setting """
# data and saving path
data_p = '/home/groups/dlmrimnd/jacob/data/combined_data'
# images to be registered, e.g., fractional anisotropy (FA) derived from DTI
R_path    = join(data_p,'warped_input_roi') 
# images to be segmented, e.g., diffusion tensor image (with six components)
S_path    = join(data_p,'warped_input_roi') 
# segmentation labels for supervised training
segm_path = join(data_p,'warped_masks_roi') 
# dense affine displacement, e.g., estimated using Elastix
affine_path = join(data_p,'deformation_fields_roi') 
# save folder for this experiment
save_path = join(data_p,'saved_results', 'restart') 
load_path = join (data_p, 'saved_results', 'run1')
if not exists(save_path):
    os.makedirs(save_path)

train_index = np.load('/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/train_index_new.npy') 
vali_index  = np.load('/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/vali_index_new.npy')

# files to save during training
# best weights of the monitor metric in checkpoint 1, i.e., seg acc
check_path  = join(save_path, 'accWeights.{epoch:02d}-{val_loss:.2f}.hdf5')
# best weights of the monitor metric in checkpoint 2, i.e., seg consistency
check2_path = join(save_path, 'consWeights.{epoch:02d}-{val_loss:.2f}.hdf5')
# begining weights for continued training

weight_path_in  = join(load_path, 'model_weight_out.h5')
# weights of the latest epoch, would overwrite the previous one
weight_path_out = join(save_path, 'model_weight_out.h5')
# training history
train_his_path  = join(save_path, "train_history.csv") 
# training history of each batch, using LossHistory_batch;
# otherwise, LossHistory_basic by default.
#batch_his_path  = join(save_path, "train_history_perBatch")

structs = ['cgc_l', 'cgc_r',  'cgh_l', 'cgh_r', 'fma', 'fmi', 'atr_l', 'atr_r', 
           'ifo_l', 'ifo_r', 'ilf_l', 'ilf_r', 'mcp', 'ml_l', 'ml_r', 'ptr_l',
           'ptr_r','str_l', 'str_r', 'unc_l', 'unc_r', 'cst_l', 'cst_r',
           'slf_l', 'slf_r']

# parameters for data generator

# params_train = {'dim_xyz': (160, 112, 128),
# params_train = {'dim_xyz': (197, 233, 189),
params_train = {'dim_xyz': (176, 80, 96),
          'R_ch': 1,
          'S_ch': 1,
          'batch_size': 1,
          # 'outputs': structs[:6],
          # int(len(params_train['outputs'])/2)
          # combined the left & right structures into one map/channel
          'n_output': 1, # nb of feature channels in the output
          'shuffle': True}
params_vali = params_train
# data Generators
train_generator = DataGenerator(**params_train).generate(train_index, 
                               R_path, S_path, segm_path, affine_path)
vali_generator = DataGenerator(**params_vali).generate(vali_index, 
                              R_path, S_path, segm_path, affine_path)

""" 
Training setting 
"""
# False: train a new model; 
# True: continue from the weights of weight_path_in, change initial_epoch accordingly
resume_pretrained = False
initial_epoch = 0 
# change here empirically, this number was used where there are n=10350 batches
# in each epoch. Thus the learning rate schedule stops the training before n_epoch.
n_epoch = 30
 
# alpha for LeakyRuLU
# Custom learning rate auto decay settings
para_decay_auto = {'initial_lr': 0.0005,  # Slightly lower initial learning rate for stability
                   'drop_percent': 0.5,   # Gradual learning rate decay
                   'patience': 12,        # Increased patience for LR reduction
                   'threshold_epsilon': 0.0,
                   'momentum': 0.9,       # Higher momentum to speed up convergence
                   'cooldown': 0,
                   'min_lr': 1e-6,        # Allow learning rate to go lower for more fine-tuning
                   'nesterov': True}

# Adam optimizer setup
opt = Adam(lr=para_decay_auto['initial_lr'], 
           beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
           decay=0.0, clipnorm=1.)

# Alpha value for LeakyReLU (optional, depending on your network architecture)
a = 0.2

# Lambda is the weight for the registration loss term
# It linearly increases from 5 to 100 during training using LossWeight (Callback),
# prioritizing segmentation early in training.
lamda = K.variable(5.) 

# Define loss functions and metrics
losses = {'warpedSrc': MeanSquaredError, 
          'movedSegm': DCLoss,    # For binary segmentation with soft Dice loss
          'srcSegm': DCLoss, 
          'nonr_def': grad_loss}  # Gradient loss for deformation field regularization

metrics = {'warpedSrc': MeanSquaredError, 
           'movedSegm': sigmoid_sftDC,  # Sigmoid + Dice Coefficient for binary segmentation
           'srcSegm': sigmoid_DC,       # Dice coefficient for source segmentation
           'nonr_def': 'mae'}           # Mean absolute error for deformation field regularization

# Loss weights for each of the tasks
# We give slightly more weight to segmentation tasks (movedSegm and srcSegm), and lower weight to nonr_def.
loss_weights = [lamda, 1.5, 1.5, 0.001*lamda]  # Registration loss is initially less weighted




""" Construct and train the model """
print("Construct model")

model = joint_model(params_train['dim_xyz'], params_train['R_ch'],
                    params_train['S_ch'], params_train['n_output'], 
                    indexing='ij',alpha=a)

Model_Summary = model.summary()

if resume_pretrained:
    model.load_weights(weight_path_in)
    print("Load pretrained model from disk!")
else:
    print('train a new model!')

if G <= 1:
    print("train with %d GPU" % G)
    parallel_model = model
# else:
#     print("train with %d GPU" % G)
#     # with tf.device("/cpu:0"):
#     parallel_model = multi_gpu_model(model, gpus=G)


parallel_model.compile(optimizer=opt, loss=losses, 
                       metrics=metrics,
                       loss_weights=loss_weights)


adptive_weight = Adaptive_LossWeight(lamda)

auto_decay = ReduceLROnPlateau(monitor='val_srcSegm_sigmoid_DC', # segmentation acc
                               factor=para_decay_auto['drop_percent'], 
                               patience=para_decay_auto['patience'], 
                               verbose=1, mode='max', 
                               epsilon=para_decay_auto['threshold_epsilon'],
                               cooldown=para_decay_auto['cooldown'], 
                               min_lr=para_decay_auto['min_lr'])


loss_history = LossHistory_basic(auto_decay)

check   = ModelCheckpoint(check_path, 
                          monitor='val_srcSegm_sigmoid_DC', # segmentation acc
                          verbose=1, save_best_only=True, 
                          save_weights_only=True, mode='max', period=1)
check_2 = ModelCheckpoint(check2_path, 
                          monitor='val_movedSegm_sigmoid_sftDC', # segmentation consistency
                          verbose=1, save_best_only=True, save_weights_only=True, 
                          mode='max', period=1)

check_eachEpoch = ModelCheckpoint(weight_path_out, monitor='val_loss', 
                                   verbose=1, save_best_only=False, 
                                   save_weights_only=True, mode='min', period=1)

csv_logger   = CSVLogger(train_his_path, separator=',', append=True)


# TensorBoard callback setup
# log_dir = join(save_path, 'logs')
# if not exists(log_dir):
#     os.makedirs(log_dir)
# tensorboard_callback = TensorBoard(log_dir=log_dir, 
#                                    histogram_freq=0,
#                                    write_graph=False, 
#                                    write_images=False)

# callbacks_list = [check, check_2, check_eachEpoch, 
#                   auto_decay, loss_history, 
#                   csv_logger, adptive_weight,
#                   tensorboard_callback]
                  
callbacks_list = [check, check_2, check_eachEpoch, 
                  auto_decay, loss_history, 
                  csv_logger, adptive_weight]

history = parallel_model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_index)//params_train['batch_size'], 
                    epochs=n_epoch,
                    verbose=1, callbacks=callbacks_list,
                    validation_data=vali_generator,
                    validation_steps=len(vali_index)//params_vali['batch_size'], 
                    initial_epoch=initial_epoch)

# Extract training and validation loss and metrics
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_warpedSrc_loss = history.history['warpedSrc_loss']
val_warpedSrc_loss = history.history['val_warpedSrc_loss']

train_movedSegm_loss = history.history['movedSegm_loss']
val_movedSegm_loss = history.history['val_movedSegm_loss']

# Similarly for other metrics
train_dice_coef = history.history['srcSegm_sigmoid_DC']
val_dice_coef = history.history['val_srcSegm_sigmoid_DC']
# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation Dice Coefficient values
plt.subplot(1, 2, 2)
plt.plot(train_dice_coef, label='Train Dice Coefficient')
plt.plot(val_dice_coef, label='Val Dice Coefficient')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()

plt.show()