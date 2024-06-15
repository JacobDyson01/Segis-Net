"""
# tensorflow/keras training for Reg-Net for non-linear 3D image registration 
developed in 
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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
from os.path import join, exists
import numpy as np
import csv
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger, EarlyStopping
#import time
from keras.optimizers import Adam
import keras.backend as K
from tools import ModelMGPU 
from Loss_metrics import grad_loss
from RegNet_model_regGener import reg_net, DataGenerator


""" Initial setting """
# data and saving path
data_p = ' '
img_path    = join(data_p,' ') # images to be registered
affine_path = join(data_p,' ') # dense affine displacement
save_path   = join(data_p,' ') # folder for this experiment
if not exists(save_path):
    os.makedirs(save_path)

train_index = np.load(' ') 
vali_index  = np.load(' ')

# files to save during training
# best weights of minimum loss value for the monitor metric
check_path  = join(save_path, 'Weights.{epoch:02d}-{val_loss:.2f}.hdf5')
# begining weights for continued training
weight_path_in = join(save_path, 'Pretrained.hdf5')
# weights of the latest epoch, would overwrite the previous one
weight_path_out = join(save_path, 'model_weight_out.h5')
# training history
train_his_path = join(save_path, "train_history.csv") 
# training history of each batch, if turned it on from LossHistory_auto
batch_his_path = join(save_path, "train_history_perBatch")


# parameters for data generator
params_train = {'dim_xyz': (112, 208, 112),
          'dim_ch': 2,
          'batch_size': 1,
          'shuffle': True}
params_vali = params_train
# data Generators
train_generator = DataGenerator(**params_train).generate(train_index, 
                               img_path, affine_path)
vali_generator = DataGenerator(**params_vali).generate(vali_index, 
                              img_path, affine_path)

""" Training setting """
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
para_decay_auto = {'initial_lr': 0.0001,
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

# loss mertrics, accuracy metrics, loss weights
losses  = {'movedFA': 'mse',
          'nonr_def': grad_loss}
metrics = {'movedFA': 'mse',
           'nonr_def':'mae'}
# weight for 'movedFA', and 'nonr_def'
loss_weights = [10., 0.1]


""" Construct and train the model """
print("Construct model")
model = reg_net(params_train['dim_xyz'], alpha=a)
# Model Summary
Model_Summary = model.summary()

if resume_pretrained:
    model.load_weights(weight_path_in)
    print("Load pretrained model from disk!")
else:
    print('train a new model!')

if G <= 1:
    print("train with %d GPU" % G)
    parallel_model = model
else:
    print("train with %d GPU" % G)
    #with tf.device("/cpu:0"):
    parallel_model = ModelMGPU(model, gpus=G)


# compile
parallel_model.compile(optimizer=opt, 
                       loss=losses, 
                       metrics=metrics, 
                       loss_weights=loss_weights)


auto_decay = ReduceLROnPlateau(monitor='val_movedFA_loss', 
                               factor=para_decay_auto['drop_percent'], 
                               patience=para_decay_auto['patience'],
                               verbose=1, mode='auto', 
                               epsilon=para_decay_auto['threshold_epsilon'], 
                               cooldown=para_decay_auto['cooldown'], 
                               min_lr=para_decay_auto['min_lr'])

class LossHistory_auto(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
#       self.accuracy = []
        self.lr = []

    def on_batch_end(self, batch, logs={}):
        # append loss of batches 
        self.losses.append(logs.get('loss'))
#        self.accuracy.append(logs.get('accuracy'))


    def on_epoch_end(self, epoch, logs={}):
        self.lr.append(auto_decay)
        print('lr: ', K.eval(self.model.optimizer.lr))
        # save loss of batches per epoch
        # remove this if not needed
        with open(batch_his_path+str(epoch)+'.csv', 'w') as f:
             wr = csv.writer(f, dialect='excel')
             wr.writerow(self.losses)

loss_history = LossHistory_auto()

early_stop = EarlyStopping(monitor='val_movedFA_loss', patience=20)

check = ModelCheckpoint(check_path, monitor='val_movedFA_loss', verbose=1, 
                             save_best_only=True, save_weights_only=True, 
                             mode='min', period=1)
check_eachEpoch = ModelCheckpoint(weight_path_out, monitor='val_loss', 
                                  verbose=1, save_best_only=False, 
                                  save_weights_only=False, mode='min', period=1)

csv_logger   = CSVLogger(train_his_path, separator=',', append=True)

callbacks_list = [check, check_eachEpoch, 
                  auto_decay, early_stop,
                  loss_history, csv_logger]

history = parallel_model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_index)//params_train['batch_size'], 
                    epochs=n_epoch, verbose=1, callbacks=callbacks_list,
                    validation_data=vali_generator,
                    validation_steps=len(vali_index)//params_vali['batch_size'], 
                    initial_epoch=initial_epoch)

