import os
from os.path import join, exists
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tools import Adaptive_LossWeight, LossHistory_basic
from Loss_metrics import sigmoid_DC, sigmoid_sftDC, grad_loss, DCLoss, MeanSquaredError
from SegisNet_model_dataGenerator import joint_model, DataGenerator

# Check if GPU is available using TensorFlow 2.x API
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available")
    try:
        # Set memory growth to avoid memory issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available")

# GPU settings
G = 1
if G == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

""" Initial setting """
# Define paths
data_p = '/home/groups/dlmrimnd/jacob/data/upgraded_segis_data'
R_path = join(data_p, 'warped_input_roi')
S_path = join(data_p, 'warped_input_roi')
segm_path = join(data_p, 'warped_masks_roi')  # Segmentation labels
affine_path = join(data_p, 'deformation_fields_roi')  # Affine transformations
save_path = join(data_p, 'saved_results', 'run_proper_5')

if not exists(save_path):
    os.makedirs(save_path)

train_index = np.load('/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/train_index_new.npy')
vali_index = np.load('/home/groups/dlmrimnd/jacob/projects/Segis-Net/code/Segis-Net/vali_index_new.npy')

# Save paths for model weights, history, etc.
check_path = join(save_path, 'accWeights.{epoch:02d}-{val_loss:.2f}.hdf5')
check2_path = join(save_path, 'consWeights.{epoch:02d}-{val_loss:.2f}.hdf5')
weight_path_in = join(save_path, 'Pretrained.hdf5')
weight_path_out = join(save_path, 'model_weight_out.h5')
train_his_path = join(save_path, "train_history.csv")

structs = ['cgc_l', 'cgc_r', 'cgh_l', 'cgh_r', 'fma', 'fmi', 'atr_l', 'atr_r',
           'ifo_l', 'ifo_r', 'ilf_l', 'ilf_r', 'mcp', 'ml_l', 'ml_r', 'ptr_l',
           'ptr_r', 'str_l', 'str_r', 'unc_l', 'unc_r', 'cst_l', 'cst_r',
           'slf_l', 'slf_r']

# Parameters for data generator
params_train = {
    'dim_xyz': (192,192,176),
    'R_ch': 1,
    'S_ch': 1,
    'batch_size': 1,
    'n_output': 1,
    'shuffle': True
}
params_vali = params_train

# Data Generators
train_generator = DataGenerator(**params_train).generate(train_index, R_path, S_path, segm_path, affine_path)
vali_generator = DataGenerator(**params_vali).generate(vali_index, R_path, S_path, segm_path, affine_path)

""" Training settings """
resume_pretrained = False
initial_epoch = 0
n_epoch = 300

# Optimizer and learning rate settings
para_decay_auto = {
    'initial_lr': 0.001,
    'drop_percent': 0.8,
    'patience': 10,
    'threshold_epsilon': 0.0,
    'momentum': 0.8,
    'cooldown': 0,
    'min_lr': 1e-7,
    'nesterov': True
}

opt = Adam(
    learning_rate=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999,
    epsilon=1e-08, decay=0.0, clipnorm=1.
)

# Lambda is the weight for registration loss term
lamda = K.variable(10.)

# Loss functions and metrics
losses = {
    'warpedSrc': MeanSquaredError,
    'movedSegm': DCLoss,
    'srcSegm': DCLoss,
    'nonr_def': grad_loss
}
metrics = {
    'warpedSrc': MeanSquaredError,
    'movedSegm': sigmoid_sftDC,
    'srcSegm': sigmoid_DC,
    'nonr_def': 'mae'
}
loss_weights = {
    'warpedSrc': lamda,
    'movedSegm': 1,
    'srcSegm': 1,
    'nonr_def': 0.01 * lamda
}

""" Construct and train the model """
print("Constructing model")

# Build the model
model = joint_model(params_train['dim_xyz'], params_train['R_ch'],
                    params_train['S_ch'], params_train['n_output'], 
                    indexing='ij', alpha=0.2)

model.summary()

if resume_pretrained:
    model.load_weights(weight_path_in)
    print("Loaded pretrained model from disk!")
else:
    print('Training a new model!')

# Compile model
model.compile(optimizer=opt, loss=losses, metrics=metrics, loss_weights=loss_weights)

# Callbacks
adaptive_weight = Adaptive_LossWeight(lamda)
auto_decay = ReduceLROnPlateau(
    monitor='val_srcSegm_sigmoid_DC', factor=para_decay_auto['drop_percent'], 
    patience=para_decay_auto['patience'], verbose=1, mode='max',
    min_delta=para_decay_auto['threshold_epsilon'], cooldown=para_decay_auto['cooldown'], 
    min_lr=para_decay_auto['min_lr']
)
loss_history = LossHistory_basic(auto_decay)

check = ModelCheckpoint(
    check_path, monitor='val_srcSegm_sigmoid_DC', verbose=1, save_best_only=True, 
    save_weights_only=True, mode='max'
)
check_2 = ModelCheckpoint(
    check2_path, monitor='val_movedSegm_sigmoid_sftDC', verbose=1, 
    save_best_only=True, save_weights_only=True, mode='max'
)
check_each_epoch = ModelCheckpoint(
    weight_path_out, monitor='val_loss', verbose=1, save_best_only=False, 
    save_weights_only=True, mode='min'
)
csv_logger = CSVLogger(train_his_path, separator=',', append=True)

callbacks_list = [check, check_2, check_each_epoch, auto_decay, loss_history, csv_logger, adaptive_weight]

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_index) // params_train['batch_size'],
    epochs=n_epoch,
    verbose=1, callbacks=callbacks_list,
    validation_data=vali_generator,
    validation_steps=len(vali_index) // params_vali['batch_size'],
    initial_epoch=initial_epoch
)
