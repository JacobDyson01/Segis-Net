"""
Tools involved in:
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021.
paper: https://arxiv.org/abs/2012.14230

Please cite the paper if the code/method would be useful to your work.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import csv
import tensorflow.keras.backend as K


class LossHistory_batch(Callback):
    """
    Callback to store the batch-wise loss history and save it after each epoch.
    
    Arguments:
    auto_decay -- The learning rate decay to apply
    batch_his_path -- The path where batch loss history should be saved
    """
    def __init__(self, auto_decay, batch_his_path):
        super(LossHistory_batch, self).__init__()
        self.auto_decay = auto_decay
        self.batch_his_path = batch_his_path
        
    def on_train_begin(self, logs=None):
        self.losses = []
        self.lr = []

    def on_batch_end(self, batch, logs=None):
        # Append batch loss to the list
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        new_lr = self.auto_decay
        self.lr.append(new_lr)
        print('lr: ', K.eval(self.model.optimizer.lr))
        # Save batch loss history at the end of each epoch
        with open(self.batch_his_path + str(epoch) + '.csv', 'w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(self.losses)


class LossHistory_basic(Callback):
    """
    Callback to store the epoch-wise loss history and print the learning rate at the end of each epoch.
    
    Arguments:
    auto_decay -- The learning rate decay to apply
    """
    def __init__(self, auto_decay):
        super(LossHistory_basic, self).__init__()
        self.auto_decay = auto_decay

    def on_train_begin(self, logs=None):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, epoch, logs=None):
        new_lr = self.auto_decay
        self.lr.append(new_lr)
        print('lr: ', K.eval(self.model.optimizer.lr))


class Adaptive_LossWeight(Callback):
    """
    Callback to adaptively change the loss weight (lambda) after each epoch.
    
    Arguments:
    lamda -- The initial lambda value
    """
    def __init__(self, lamda, **kwargs):
        self.lamda = lamda
        super(Adaptive_LossWeight, self).__init__()

    def on_train_begin(self, logs=None):
        self.lamdas = []

    def on_epoch_end(self, epoch, logs=None):
        self.lamdas.append(self.lamda)

        old_lamda = float(K.get_value(self.lamda))
        logs['lambda'] = old_lamda

        if old_lamda < 100:
            # Adjust lambda linearly
            new_lamda = old_lamda + 0  # Change this to a value if needed

            K.set_value(self.lamda, new_lamda)

        print('Loss weight lambda used: ', K.get_value(self.lamda))
