"""
# Reg-Net model for non-linear 3D image registration, developed in:
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021.
paper: https://arxiv.org/abs/2012.14230

please cite the paper if the code/method would be useful to your work.

# for suggestions and questions, contact: BL (b.li@erasmusmc.nl)
"""

from keras.models import Model
from keras.layers import Input, concatenate,Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Add
import tensorflow as tf
# import keras.backend as K
#from keras.utils.training_utils import multi_gpu_model
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
#from keras.engine import Layer
import numpy as np
from os.path import join
from nilearn import image
import nibabel as nib
from Transform_layer_interpn_0 import SpatialTransformer


def ConvBlockA(x, ch_1, ch_2, alpha):

    out = x    
    for i in [ch_1, ch_2]:
        out = Conv3D(i, (3, 3, 3),
                            use_bias=False, padding='same')(out) # activation=None,
        out = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(out)
        out = LeakyReLU(alpha=alpha)(out)               

    return out

def ConvBlockB(x, ch_1, ch_2, ch_3, alpha):

    out = x    
    for i in [ch_1, ch_2, ch_3]:
        out = Conv3D(i, (3, 3, 3),
                            use_bias=False, padding='same')(out) # activation=None,
        out = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(out)
        out = LeakyReLU(alpha=alpha)(out) 
        
    return out


def reg_net(img_xyz, alpha=0.2):
    num_start_ch = 16
    alpha = alpha
    """input layers"""
    tgt = Input(shape=(img_xyz) + (1,))  # Single channel for target image
    src = Input(shape=(img_xyz) + (1,))  # Single channel for source image
    # tgt = Input(shape=(img_xyz))  # Single channel for target image
    # src = Input(shape=(img_xyz))  # Single channel for source image
    aff_warped = Input(shape=(img_xyz) + (1,))  # 3 channels for deformation field  
    

    
   

    """online affine-warp"""
    # aff_warped = SpatialTransformer(interp_method='linear', indexing='ij', 
    #                                 name='aff_warped')([src, aff_def]) 
    # print('Warped Image - min:', tf.reduce_min(aff_warped).numpy(), 'max:', tf.reduce_max(aff_warped).numpy())
    # aff_warped = tf.clip_by_value(aff_warped, 0, 255)
    inputs = concatenate([tgt, aff_warped], axis=-1)

    """Encoder"""
    conv_1 = ConvBlockA(inputs, int(num_start_ch / 2), num_start_ch, alpha)
    pool_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_1)

    conv_2 = ConvBlockA(pool_1, num_start_ch*2, num_start_ch*2, alpha)
    pool_2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_2)

    conv_3 = ConvBlockA(pool_2, num_start_ch*4, num_start_ch*4, alpha)
    pool_3 = MaxPooling3D(pool_size=(2, 2, 2))(conv_3)

    conv_4 = ConvBlockA(pool_3, num_start_ch*8, num_start_ch*8, alpha)
    pool_4 = MaxPooling3D(pool_size=(2, 2, 2))(conv_4)

    """fifth layer"""
    conv_5 = ConvBlockB(pool_4, num_start_ch*16, num_start_ch*8, num_start_ch*8, alpha)
    up_6 = UpSampling3D(size=(2, 2, 2))(conv_5)

    """Decoder"""
    up_6 = concatenate([up_6, conv_4], axis=4)
    conv_6 = ConvBlockB(up_6, num_start_ch*8, num_start_ch*4, num_start_ch*4, alpha)
    up_7 = UpSampling3D(size=(2, 2, 2))(conv_6)

    up_7 = concatenate([up_7, conv_3], axis=4)
    conv_7 = ConvBlockB(up_7, num_start_ch*4, num_start_ch*2, num_start_ch*2, alpha)
    up_8 = UpSampling3D(size=(2, 2, 2))(conv_7)

    up_8 = concatenate([up_8, conv_2], axis=4)
    conv_8 = ConvBlockB(up_8, num_start_ch*2, num_start_ch, num_start_ch, alpha)
    up_9 = UpSampling3D(size=(2, 2, 2))(conv_8)

    up_9 = concatenate([up_9, conv_1], axis=4)
    conv_9 = ConvBlockA(up_9, num_start_ch, int(num_start_ch / 2), alpha)

    """shape the nonr_def"""
    nonr_def = Conv3D(3, (3, 3, 3), activation=None, padding='same', name='nonr_def',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv_9)
    

    """composite transformation"""
    # all_def = Add()([nonr_def, aff_def])
    

    """image warp use the composite transformation"""
    y = SpatialTransformer(interp_method='linear', indexing='ij', name='movedFA')([src, nonr_def])
    

    """output"""
    model = Model(inputs=[tgt, src, aff_warped], outputs=[y, nonr_def])

    return model


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_xyz, dim_ch, batch_size, shuffle=True):
      'Initialization'
      self.dim_xyz = dim_xyz  # image size
      # nb of feature channels in the images to be registered, e.g., 1
      self.dim_ch = dim_ch      
      self.batch_size = batch_size 
      self.shuffle = shuffle    # random shuffle the order of data/batch  

  # if training and validation data save in same directory, no need to use part_path (e.g., train_p or vali_p). 
  def generate(self, part_index, img_path, affine_path):
      # part_index: list of train or validation-samples.     
      # img_path: images for registration
      # affine alignment is a preprocessing step in the method
      # we use as input the original image and the affine matrix, rather than affine-warped image-pair.
      # affine matrix obtained using Elastix (Klein et al., TMI 2010) is stored in affine_path
      
      'Generates batches of samples'
      
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(part_index)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [part_index[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              x, y = self.__data_generation(list_IDs_temp, img_path, affine_path)

              yield x, y

  def __get_exploration_order(self, part_index):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(part_index))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp, img_path, affine_path):
      'Generates data of batch_size samples' 
      # Initialization
      tgt = np.zeros((self.batch_size, *self.dim_xyz, self.dim_ch)).astype(dtype='float32')
      src = np.zeros((self.batch_size, *self.dim_xyz, self.dim_ch)).astype(dtype='float32')
      # keras.backend.floatx() default is
      # Out[2]: 'float32'
      
      # 3 channels because of 3D images to be registered
      zeros   = np.zeros((self.batch_size, *self.dim_xyz, 3)).astype(dtype='float16')
      aff_def = np.zeros((self.batch_size, *self.dim_xyz, 3)).astype(dtype='float32')
      # Generate batch
      for i, ID in enumerate(list_IDs_temp):
          # IDs of the images to be registered, ID1: target, ID2: source
          # for instance, ID=[sub_001, sub_002], then:
          ID1 = ID[0]
          ID2 = ID[1]

          # img to be registered
          # exampled of loading nifty image and numpy array are provided as belows
          # nfity
          tgt_p   = join(img_path, str(ID1), 'tgt_FA.nii.gz')
          src_p   = join(img_path, str(ID2), 'src_FA.nii.gz')
          tgt_img = image.load_img(tgt_p).get_fdata().astype(dtype='float32')
          src_img = image.load_img(src_p).get_fdata().astype(dtype='float32')
          # zipped numpy array
#          tgt_img   = np.load(join(img_path, str(ID1), 'file_name.npz'))['img']

          tgt[i, :, :, :, 0]       = tgt_img
          src[i, :, :, :, 0]       = src_img
          
          # pre-estimated dense affine (displacement) map, size: (x,y,z,3)
          affine = np.load(join(affine_path,str(ID2)+'.'+str(ID1),'deformationField.npz'))['deff']
          aff_def[i,...] = affine
          # note that the map is in ijk-index rather than the world coordicate
          # the step2 in function apply_affine_deff_from_Elastix is an example to convert 
          # deformation obtained from Elastix to numpy array         

      return [tgt,src,aff_def], [tgt,zeros]
  
    
def apply_affine_deff_from_Elastix(deff_file, tgt_p, src_p):

    # step1: load deformation field from elastix output
    deff = image.load_img(deff_file).get_fdata()
    deff = deff[:,:,:,0,:].astype('float32')

    # step2: shift affine matrix to dense displacement
    fixed_aff =nib.load(tgt_p).affine
    moving_aff=nib.load(src_p).affine
    
    src = image.load_img(src_p).get_fdata()
    diff_aff  = fixed_aff - moving_aff

    deff[...,0] = -deff[...,0] + diff_aff[0,3]
    deff[...,1] = -deff[...,1] + diff_aff[1,3]   
    deff[...,2] =  deff[...,2] + diff_aff[2,3]

    # step3: warp the src file with the displacement   
#    sess = tf.InteractiveSession()
    # https://elastix.lumc.nl/doxygen/classelastix_1_1LinearInterpolator.html
#    warped = interpn(src, loc, interp_method='linear').eval() 
    src_tensor = tf.cast(tf.convert_to_tensor(src[np.newaxis,...,np.newaxis]),'float32')
    deff_tensor = tf.cast(tf.convert_to_tensor(deff[np.newaxis,...]),'float32')
    warped = SpatialTransformer(interp_method='linear', indexing='ij', 
                                name='warped_img')([src_tensor, deff_tensor]).eval()

    return warped


#class Softmax(Layer):
#    def __init__(self, axis=-1, **kwargs):
#        self.axis=axis
#        super(Softmax, self).__init__(**kwargs)
#
#    def build(self,input_shape):
#        pass
#
#    def call(self, x, mask=None):
#        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
#        s = K.sum(e, axis=self.axis, keepdims=True)
#        return e / s
#
#    def compute_output_shape_for(self, input_shape):
#        return input_shape
#
#
#class Deconvolution3D(Layer):
#    def __init__(self, nb_filter, kernel_dims, output_shape, subsample):
#        self.nb_filter = nb_filter
#        self.kernel_dims = kernel_dims
#        self.strides = (1,) + subsample + (1,)
#        self.output_shape_ = output_shape
#        assert K.backend() == 'tensorflow'
#        super(Deconvolution3D, self).__init__()
#
#    def build(self, input_shape):
#        assert len(input_shape) == 5
#        self.input_shape_ = input_shape
#        W_shape = self.kernel_dims + (self.nb_filter, input_shape[4],)
#
#
#        self.b = self.add_weight((1, 1, 1, self.nb_filter,),
#                                 initializer='zero', name='{}_b'.format(self.name))
#
#        self.W = self.add_weight(name='{}_W'.format(self.name),
#                                 shape=W_shape,
#                                 initializer='uniform',
#                                 trainable=True)
#
#        super(Deconvolution3D, self).build(input_shape)  # Be sure to call this somewhere!
#
#    #     def get_output_shape_for(self, input_shape):
#    def compute_output_shape(self, input_shape):
#        return (None,) + self.output_shape_[1:]
#
#
#    def call(self, x, mask=None):
#        return tf.nn.conv3d_transpose(x, self.W, output_shape=self.output_shape_,
#                                      strides=self.strides, padding='SAME', name=self.name) + self.b
#
#    def get_config(self):
#        base_config = super(Deconvolution3D, self).get_config().copy()
#        base_config['output_shape'] = self.output_shape_
#        return base_config   