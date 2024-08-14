"""
Sptial Transformer Layer, and diffusion regularzation metric (Grad), adopted from
  Dalca et al., MICCAI 2018.
  
In Segis-Net (Li et al., NeuroImage 2021), we made changes to the interpn function 
in line #226, #230, #273 to replace clip by max location with clip by zeros. 

"""

import numpy as np
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import itertools



class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer, used in:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/tf/layers.py
    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')
        
        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1]. 
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        self.is_affine = len(trf_shape) == 1 or \
                         (len(trf_shape) == 2 and all([f == (self.ndims+1) for f in trf_shape]))

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d' 
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # go from affine
        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0,:]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:  # go from vector to matrix
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        # note this is unnecessarily extra graph since at every batch entry we have a tf.eye graph
#      TODO  trf += tf.eye(self.ndims+1)[:self.ndims,:]  # add identity, hence affine is a shift from identitiy
        trf += tf.eye(self.ndims+1) 
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        vol, trf = inputs
        vol_shape = tf.shape(vol)[:-1]
        
        # Clip trf values to stay within valid indices
        # trf = tf.clip_by_value(trf, 0, tf.cast(vol_shape - 1, trf.dtype))
        
        return transform(vol, trf, interp_method=self.interp_method)


class Grad():
    """
    N-D gradient loss
    https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/tf/losses.py
    """

    def __init__(self, penalty='l2'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)

# the functions below are used by the Spatial Transformer
def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow
    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions
    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'
    Returns:
        new interpolated volume of the same size as the entries in loc
    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """
    
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    # since loc can be a list, nb_dims has to be based on vol.
    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')

    # interpolate
    if interp_method == 'linear':
        loc0 = tf.floor(loc)

        # clip values
        # --- changed Jun22,2020 by BL, want zero instead of max_loc, remove clip
        max_loc = [d - 1 for d in vol.get_shape().as_list()]
#        loc0lst = [tf.clip_by_value(loc0[...,d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [loc0[...,d] for d in range(nb_dims)]
        
        # get other end of point cube
#        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        loc1 = [loc0lst[d] + 1 for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]
        # ------end of change
        
        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - loc[...,d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0] # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector 
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        
        for c in cube_pts:
            
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            idx = sub2ind(vol.shape[:-1], subs)

            vol_val = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]

            wt = prod_n(wts_lst)
            wt = K.expand_dims(wt, -1)
            
            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val
#            print('interp shape', interp_vol.shape)
        
    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')
#        roundloc = tf.cast(tf.math.floor(loc), 'int32')
        # clip values
#        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        # --- changed Jun22,2020 by BL, want zero instead of max_loc
        # Note that on CPU, if an out of bound index is found, an error is returned. 
        # On GPU, if an out of bound index is found, a 0 is stored in the corresponding output value.
#        roundloc = [tf.clip_by_value(roundloc[...,d], 0, max_loc[d]) for d in range(nb_dims)]
        roundloc = [roundloc[...,d] for d in range(nb_dims)]
        # --- end of change
        
        # get values
        # tf stacking is slow. replace with gather
#        roundloc = tf.stack(roundloc, axis=-1)
#        interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx) 

    return interp_vol

def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx

def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow
    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid
    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)
    Returns:
        shift field (Tensor) of size *volshape x N
    TODO: 
        allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """

    if isinstance(volshape, (tf.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()
    
    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)) :
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  
    mesh = [tf.cast(f, 'float32') for f in mesh]
    
    if shift_center:
        mesh = [mesh[f] - (volshape[f]-1)/2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)

def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size
    Parameters:
        volshape: the volume size
        **args: "name" (optional)
    Returns:
        A list of Tensors
    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """
    
    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)

def flatten(v):
    """
    flatten Tensor v
    
    Parameters:
        v: Tensor to be flattened
    
    Returns:
        flat Tensor
    """

    return tf.reshape(v, [-1])

def meshgrid(*args, **kwargs):
    """
    
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921
    
    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """
# TODO pop or get
    indexing = kwargs.pop("indexing", "xy")
    name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype

    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow. 
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # TODO(nolivia): improve performance with a broadcast  
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):       
        output[i] = tf.tile(output[i], tf.stack([*sz[:i], 1, *sz[(i+1):]]))
    return output

def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow
    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.
    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
    
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    
    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes
    if isinstance(loc_shift.shape, (tf.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
#    loc_shift = tf.cast(loc_shift,'float32')
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return interpn(vol, loc, interp_method=interp_method)    