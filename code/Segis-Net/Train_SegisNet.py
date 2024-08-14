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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
#import time
from keras.optimizers import Adam
import keras.backend as K
from tools import Adaptive_LossWeight, LossHistory_basic
from Loss_metrics import sigmoid_DC, sigmoid_sftDC, grad_loss, DCLoss, MeanSquaredError
from SegisNet_model_dataGenerator import joint_model, DataGenerator


""" Initial setting """
# data and saving path
data_p = '/home/groups/dlmrimnd/jacob/data/'
# images to be registered, e.g., fractional anisotropy (FA) derived from DTI
R_path    = join(data_p,'input_images_roi') 
# images to be segmented, e.g., diffusion tensor image (with six components)
S_path    = join(data_p,'input_images_roi') 
# segmentation labels for supervised training
segm_path = join(data_p,'binary_masks_roi') 
# dense affine displacement, e.g., estimated using Elastix
affine_path = join(data_p,'deformation_fields_roi') 
# save folder for this experiment
save_path   = join(data_p,'results') 
if not exists(save_path):
    os.makedirs(save_path)

train_index = np.load('/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/train_index.npy') 
vali_index  = np.load('/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/vali_index.npy')

# files to save during training
# best weights of the monitor metric in checkpoint 1, i.e., seg acc
check_path  = join(save_path, 'accWeights.{epoch:02d}-{val_loss:.2f}.hdf5')
# best weights of the monitor metric in checkpoint 2, i.e., seg consistency
check2_path = join(save_path, 'consWeights.{epoch:02d}-{val_loss:.2f}.hdf5')
# begining weights for continued training
weight_path_in  = join(save_path, 'Pretrained.hdf5')
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
params_train = {'dim_xyz': (160, 160, 160),
          'R_ch': 1,
          'S_ch': 1,
          'batch_size': 1,
          # 'outputs': structs[:6],
          # int(len(params_train['outputs'])/2)
          # combined the left & right structures into one map/channel
          'n_output': 3, # nb of feature channels in the output
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
n_epoch = 300 
 
# alpha for LeakyRuLU
a=0.2

## custom learning rate auto decay
# the inital_lr will decrease to lr*drop_percent, once the monitor metric
# stops improving for patience epochs.
para_decay_auto = {'initial_lr': 0.001,
                   'drop_percent': 0.8,
                   'patience': 10,
                   'threshold_epsilon': 0.0,
                   'momentum': 0.8,
                   'cooldown': 0,
                   'min_lr': 1e-7,
                   'nesterov':True}

# default Adam setting, except initial LR, and clipnorm
opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, 
           epsilon=1e-08, decay=0.0, clipnorm=1.)
# alpha for LeakyRuLU
a=0.2

"""
lambda is the weight for registration loss term,
which linearly increased from 10 to 100 during training using LossWeight(Callback),
to initialize the model with a focus on the segmengtation task
"""
lamda= K.variable(10.) 

# loss mertrics, accuracy metrics, loss weights
losses  = {'warpedSrc':MeanSquaredError, 
           'movedSegm':DCLoss, # probabilistic
           'srcSegm':DCLoss, 
           'nonr_def':grad_loss} 
metrics = {'warpedSrc':MeanSquaredError, 
           'movedSegm':sigmoid_sftDC, # binary
           'srcSegm':sigmoid_DC, # binary
           'nonr_def':'mae'} 

loss_weights = [lamda, 1, 1, 0.01*lamda]



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
#else:
    #print("train with %d GPU" % G)
    #with tf.device("/cpu:0"):
    #parallel_model = ModelMGPU(model, gpus=G)


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

