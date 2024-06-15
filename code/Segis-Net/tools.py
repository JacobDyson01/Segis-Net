
"""
tools envolved in:
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021.
paper: https://arxiv.org/abs/2012.14230

please cite the paper if the code/method would be useful to your work.

# for suggestions and questions, contact: BL (b.li@erasmusmc.nl)
"""
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import Callback
import csv
import keras.backend as K
#import numpy as np
#from os.path import join
#from nilearn import image


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        # if 'load' in attrname or 'save' in attrname:
        if 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class LossHistory_batch(Callback):
    """Arguments: auto_decay, batch_his_path"""
    
    def __init__(self, auto_decay, batch_his_path):
        super(LossHistory_batch, self).__init__()
        self.auto_decay = auto_decay
        self.batch_his_path = batch_his_path
        
    def on_train_begin(self, logs={}):
        self.losses = []
#       self.accuracy = []
        self.lr = []

    def on_batch_end(self, batch, logs={}):
        # append loss of batches 
        self.losses.append(logs.get('loss'))
#        self.accuracy.append(logs.get('accuracy'))


    def on_epoch_end(self, epoch, logs={}):
        new_lr = self.auto_decay
        self.lr.append(new_lr)
        print('lr: ', K.eval(self.model.optimizer.lr))
        # save loss of batches per epoch
        # remove this if not needed
        with open(self.batch_his_path+str(epoch)+'.csv', 'w') as f:
             wr = csv.writer(f, dialect='excel')
             wr.writerow(self.losses)


class LossHistory_basic(Callback):
    """Arguments: auto_decay"""
    
    def __init__(self, auto_decay):
        super(LossHistory_batch, self).__init__()
        self.auto_decay = auto_decay
        
    def on_train_begin(self, logs={}):
        self.losses = []
#       self.accuracy = []
        self.lr = []

    def on_epoch_end(self, epoch, logs={}):
        new_lr = self.auto_decay
        self.lr.append(new_lr)
        print('lr: ', K.eval(self.model.optimizer.lr))
             
             
class Adaptive_LossWeight(Callback):
    """Arguments: initial lamda"""
    def __init__(self, lamda,**kwargs):
        self.lamda = lamda
#        self.beta = beta
        super(Adaptive_LossWeight, self).__init__()
    def on_train_begin(self, logs={}):
        self.lamdas = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.lamdas.append(self.lamda) 

        old_lamda = float(K.get_value(self.lamda))
        logs['lambda'] = old_lamda

        if old_lamda < 100:
            # nonlinear change
#            new_lamda = old_lamda+ 0.1*epoch 
            
            # linear change
            new_lamda = old_lamda+ 4

            K.set_value(self.lamda, new_lamda)

        print('loss weight lamda used: ', K.get_value(self.lamda))
             