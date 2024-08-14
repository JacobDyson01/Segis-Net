"""
tensorflow/keras-based Segis-Net model for concurrent registration and 
segmentation of multiple structures, developed in 
    Li et al., Longitudinal diffusion MRI analysis using Segis-Net: a single-step deep-learning
    framework for simultaneous segmentation and registration. NeuroImage 2021.
paper: https://arxiv.org/abs/2012.14230

please cite the paper if the code/method would be useful to your work.

# for suggestions and questions, contact: BL (b.li@erasmusmc.nl)
"""

from keras.models import Model
from keras.layers import Input, Add,Lambda
import keras.backend as K
#from keras import losses
import tensorflow as tf
#from keras.utils.training_utils import multi_gpu_model
#from keras.initializers import RandomNormal
#from keras.layers.advanced_activations import LeakyReLU
from keras.engine import Layer
import numpy as np
from os.path import join
from nilearn import image
import nibabel as nib
from SegNet_model_segGener import seg_net
from RegNet_model_regGener import reg_net
from Transform_layer_interpn_0 import SpatialTransformer as Transformer


class Softmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis=axis
        super(Softmax, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape_for(self, input_shape):
        return input_shape

class Deconvolution3D(Layer):
    def __init__(self, nb_filter, kernel_dims, output_shape, subsample):
        self.nb_filter = nb_filter
        self.kernel_dims = kernel_dims
        self.strides = (1,) + subsample + (1,)
        self.output_shape_ = output_shape
        assert K.backend() == 'tensorflow'
        super(Deconvolution3D, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 5
        self.input_shape_ = input_shape
        W_shape = self.kernel_dims + (self.nb_filter, input_shape[4],)


        self.b = self.add_weight((1, 1, 1, self.nb_filter,),
                                 initializer='zero', name='{}_b'.format(self.name))

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=W_shape,
                                 initializer='uniform',
                                 trainable=True)

        super(Deconvolution3D, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (None,) + self.output_shape_[1:]

    def call(self, x, mask=None):
        return tf.nn.conv3d_transpose(x, self.W, output_shape=self.output_shape_,
                                      strides=self.strides, padding='SAME', name=self.name) + self.b

    def get_config(self):
        base_config = super(Deconvolution3D, self).get_config().copy()
        base_config['output_shape'] = self.output_shape_
        return base_config


def joint_model(img_xyz, R_ch, S_ch, n_output, indexing='ij', alpha=0.2):
    indexing = indexing
    alpha = alpha
    # inputs
    tgt = Input(shape=img_xyz + (1,))  # Single channel
    src = Input(shape=img_xyz + (1,))  # Single channel
    S_src = Input(shape=img_xyz + (1,))
    

    # tgt = Input(shape=img_xyz)  # Single channel
    # src = Input(shape=img_xyz)  # Single channel
    # S_src = Input(shape=img_xyz)
    aff_def = Input(shape=img_xyz + (3,))

    # seg_model = seg_net(img_xyz, S_ch, n_output, alpha=alpha)
    seg_model = seg_net(img_xyz, S_ch, n_output, alpha=alpha)
    reg_model = reg_net(img_xyz, alpha=alpha)

    # load seg-net and reg-net    
    src_segm = seg_model(S_src)
    print(f'src_segm shape: {src_segm.shape}')

    [y, nonr_def] = reg_model([tgt, src, aff_def])
    print(f'y shape: {y.shape}')
    print(f'nonr_def shape: {nonr_def.shape}')

    # name layer for output-specific loss    
    src_segm = Lambda(lambda x: x, name='srcSegm')(src_segm)
    y = Lambda(lambda x: x, name='warpedSrc')(y)
    nonr_def = Lambda(lambda x: x, name='nonr_def')(nonr_def)

    # composite deformation of affine and non-linear
    all_def = Add()([nonr_def, aff_def])
    print(f'all_def shape: {all_def.shape}')

    # warp source segmentation        
    tgt_segm = Transformer(interp_method='linear', indexing=indexing, name='movedSegm')([src_segm, all_def])
    print(f'tgt_segm shape: {tgt_segm.shape}')

    model = Model(inputs=[tgt, src, S_src, aff_def], outputs=[y, tgt_segm, src_segm, nonr_def])

    return model


class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_xyz, R_ch, S_ch, batch_size, n_output, shuffle=True):
      'Initialization'
      self.dim_xyz = dim_xyz
      # nb of feature channels in the images to be registered, e.g., 1 for Segis-Net   
      self.R_ch = R_ch
      # nb of feature channels in the images to be segmented, e.g., 6 
      self.S_ch = S_ch       
      self.batch_size = batch_size
      # nb of channels/structures in the segmentation, e.g., 3 
      self.n_output = n_output      
      self.shuffle = shuffle

  def generate(self, part_index, R_path, S_path, segm_path, affine_path):
      # part_index: list of train or validation-samples.       
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
              x, y = self.__data_generation(list_IDs_temp, R_path, S_path, segm_path, affine_path)

              yield x, y

  def __get_exploration_order(self, part_index):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(part_index))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp, R_path, S_path, segm_path, affine_path):
      'Generates data of batch_size samples'
      # Initialization
      R_tgt = np.zeros((self.batch_size, *self.dim_xyz, self.R_ch)).astype(dtype='float32')
      R_src = np.zeros((self.batch_size, *self.dim_xyz, self.R_ch)).astype(dtype='float32')
      # R_tgt = np.zeros((self.batch_size, *self.dim_xyz)).astype(dtype='float32')
      # R_src = np.zeros((self.batch_size, *self.dim_xyz)).astype(dtype='float32')
      # keras.backend.floatx() default is
      # Out[2]: 'float32'
      S_src = np.zeros((self.batch_size, *self.dim_xyz, self.S_ch)).astype(dtype='float32')
      # S_src = np.zeros((self.batch_size, *self.dim_xyz)).astype(dtype='float32')
      segm_tgt = np.zeros((self.batch_size, *self.dim_xyz, self.n_output)).astype(dtype='int8')
      segm_src = np.zeros((self.batch_size, *self.dim_xyz, self.n_output)).astype(dtype='int8')
    #   segm_tgt = np.zeros((self.batch_size, *self.dim_xyz)).astype(dtype='int8')
    #   segm_src = np.zeros((self.batch_size, *self.dim_xyz)).astype(dtype='int8')
      
      # the displacement has 3 channels, because of 3-dimention
      zeros   = np.zeros((self.batch_size, *self.dim_xyz, 3)).astype(dtype='float16')
      aff_def = np.zeros((self.batch_size, *self.dim_xyz, 3)).astype(dtype='float32')
      # Generate batch
      for i, ID in enumerate(list_IDs_temp):
    # IDs of the images to be registered, ID1: target, ID2: moving
    # Extract the subject and session information from the directory name
        subject_ses = ID.split('_')
        subject = subject_ses[0]
        ses1 = subject_ses[1]
        ses2 = subject_ses[2]

    # Define paths to the target and source images
        tgt_p = join(R_path, f'{subject}_{ses1}_{ses2}', 'target_roi.nii.gz')
        src_p = join(R_path, f'{subject}_{ses1}_{ses2}', 'source_roi.nii.gz')
        # print(f'{tgt_p}')
        tgt_img = nib.load(tgt_p).get_fdata().astype(dtype='float32')
        src_img = nib.load(src_p).get_fdata().astype(dtype='float32')
        

    # Load the source image to be segmented and the label
        tensor_p = join(S_path, f'{subject}_{ses1}_{ses2}', 'source_roi.nii.gz')
        tensor = nib.load(tensor_p).get_fdata().astype(dtype='float32')

        segm1_p = join(segm_path, f'{subject}_{ses1}', 'binary_mask_roi.nii.gz')
        segm1 = nib.load(segm1_p).get_fdata().astype(dtype='int8')

        segm2_p = join(segm_path, f'{subject}_{ses2}', 'binary_mask_roi.nii.gz')
        segm2 = nib.load(segm2_p).get_fdata().astype(dtype='int8')
            
          #add data-augmentation here, if needed
          
          # intensity normalization within the brain tissue (zero mean, std one), if needed 
        tensor -= np.mean(tensor)
        tensor /= np.std(tensor)

        tensor = self.resize_image(tensor, (self.dim_xyz))
        # segm1 = self.resize_image(segm1, (self.batch_size, *self.dim_xyz, self.n_output))
        # segm2 = self.resize_image(segm2, (self.batch_size, *self.dim_xyz, self.n_output))
        # # segm1 = self.resize_image(segm1, (*self.dim_xyz))
        # # segm2 = self.resize_image(segm2, (*self.dim_xyz))
        tgt_img = self.resize_image(tgt_img, (self.dim_xyz))
        src_img = self.resize_image(src_img, (self.dim_xyz))
        # tensor = self.resize_image(tensor, (*self.dim_xyz, self.S_ch))
        tensor = np.expand_dims(tensor, axis=-1)
        # segm1 = self.resize_image(segm1, (*self.dim_xyz, self.n_output))
        # segm2 = self.resize_image(segm2, (*self.dim_xyz, self.n_output))

        
        segm1 = self.resize_image(segm1, self.dim_xyz)
        segm2 = self.resize_image(segm2, self.dim_xyz)
        segm1 = np.expand_dims(segm1, axis=-1)
        segm2 = np.expand_dims(segm2, axis=-1)
        # tgt_img = self.resize_image(tgt_img, (*self.dim_xyz, self.R_ch))
        # src_img = self.resize_image(src_img, (*self.dim_xyz, self.R_ch))
        tgt_img = np.expand_dims(tgt_img, axis=-1)
        src_img = np.expand_dims(src_img, axis=-1)

        # pre-estimated dense affine (displacement) map, size: (x,y,z,3)
        affine_p = join(affine_path, f'{subject}_{ses1}_{ses2}', 'deformationField.nii.gz')
        affine = nib.load(affine_p).get_fdata().astype(dtype='float32')
        
        if affine.ndim == 5:
                affine = affine.squeeze()

        # affine = self.resize_image(affine, (self.batch_size, *self.dim_xyz, 3))
        affine = self.resize_image(affine, (*self.dim_xyz, 3))

        # Normalize affine deformation field values
        affine -= np.min(affine)
        affine /= np.max(affine)
        affine *= np.array(self.dim_xyz) - 1
        affine = np.clip(affine, 0, np.array(self.dim_xyz) - 1)
        
        
        print(f'affine min: {np.min(affine)}, max: {np.max(affine)}')


        print(f"tensor shape: {tensor.shape}")
        print(f"segm1 shape: {segm1.shape}")
        print(f"segm2 shape: {segm2.shape}")
        print(f"tgt_img shape: {tgt_img.shape}")
        print(f"src_img shape: {src_img.shape}")
        print(f"aff_def shape: {affine.shape}")
        S_src[i] = tensor
        R_tgt[i] = tgt_img
        R_src[i] = src_img
        segm_tgt[i] = segm1
        segm_src[i] = segm2
        
          
          
					# Assign the affine deformation field to aff_def
        aff_def[i] = affine
          # note that the map is in ijk-index rather than the world coordicate
          # the step2 in function apply_affine_deff_from_Elastix is an example to convert 
          # deformation obtained from Elastix to numpy array

      return [R_tgt,R_src,S_src,aff_def], [R_tgt,segm_tgt,segm_src,zeros]

  def resize_image(self, image, target_shape):
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target_shape, image.shape)]
        return zoom(image, factors, order=1)


# def apply_affine_deff_from_Elastix(deff_file, tgt_p, src_p):

#     # step1: load deformation field from elastix output
#     deff = image.load_img(deff_file).get_fdata()
#     deff = deff[:,:,:,0,:].astype('float32')

#     # step2: shift affine matrix to dense displacement
#     fixed_aff =nib.load(tgt_p).affine
#     moving_aff=nib.load(src_p).affine
    
#     src = image.load_img(src_p).get_fdata()
#     diff_aff  = fixed_aff - moving_aff

#     deff[...,0] = -deff[...,0] + diff_aff[0,3]
#     deff[...,1] = -deff[...,1] + diff_aff[1,3]   
#     deff[...,2] =  deff[...,2] + diff_aff[2,3]

#     # step3: warp the src file with the displacement   
# #    sess = tf.InteractiveSession()
#     # https://elastix.lumc.nl/doxygen/classelastix_1_1LinearInterpolator.html
# #    warped = interpn(src, loc, interp_method='linear').eval() 
#     src_tensor = tf.cast(tf.convert_to_tensor(src[np.newaxis,...,np.newaxis]),'float32')
#     deff_tensor = tf.cast(tf.convert_to_tensor(deff[np.newaxis,...]),'float32')
#     warped = Transformer(interp_method='linear', indexing='ij', 
#                                 name='warped_img')([src_tensor, deff_tensor]).eval()

#     return warped

# #def transformer(img_xyz, img_ch,indexing='ij', interp='nearest'):
# #    """
# #    interp_method: 'linear' or 'nearest'
# #    """
# #    
# #    deff      = Input(shape=(*img_xyz)+(3,)) 
# #    src_test  = Input(shape=(*img_xyz, img_ch))
# #  
# #    tgt_test = Transformer(interp_method=interp, indexing=indexing, name='test_moved')([src_test, deff])
# #
# #    model = Model(inputs=[src_test, deff], outputs=tgt_test)
# #    return model