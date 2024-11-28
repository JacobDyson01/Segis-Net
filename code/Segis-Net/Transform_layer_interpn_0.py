"""
Spatial Transformer Layer and diffusion regularization metric (Grad),
adopted from Dalca et al., MICCAI 2018.
 
In Segis-Net (Li et al., NeuroImage 2021), changes were made to the `interpn` function
in lines #226, #230, #273 to replace clipping by max location with clipping by zeros.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import itertools

class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer TensorFlow / Keras Layer, used in:
    - Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca et al., MICCAI 2018.
    https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/tf/layers.py

    This layer applies affine or non-linear transformations to images.
    """

    def __init__(self, interp_method='linear', indexing='ij', single_transform=False, **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest' (interpolation method)
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
            single_transform: whether a single transform is supplied for the whole batch
        """
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2.'
                             'First argument is the image, second is the transform.')
        
        # Set up number of dimensions and input shapes
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape

        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # Check if the transformation is affine or non-affine
        self.is_affine = len(trf_shape) == 1 or \
                         (len(trf_shape) == 2 and all([f == (self.ndims + 1) for f in trf_shape]))

        # Check shape compatibility for affine transformations
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise ValueError('Expected flattened affine of length %d but got %d' % (ex, trf_shape[0]))

        if not self.is_affine and trf_shape[-1] != self.ndims:
            raise ValueError('Offset flow field size expected: %d, found: %d' % (self.ndims, trf_shape[-1]))

        # Mark the layer as built
        self.built = True

    def call(self, inputs):
        vol, trf = inputs
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        if self.indexing == 'xy':
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        trf += tf.eye(self.ndims + 1)
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        vol, trf = inputs
        return transform(vol, trf, interp_method=self.interp_method)

class Grad:
    """
    N-D gradient loss used for regularization.
    """

    def __init__(self, penalty='l2'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)

# Helper functions
def interpn(vol, loc, interp_method='linear'):
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    nb_dims = loc.shape[-1]

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    loc = tf.cast(loc, 'float32')

    if interp_method == 'linear':
        loc0 = tf.floor(loc)
        max_loc = [d - 1 for d in vol.get_shape().as_list()[:-1]]

        loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        diff_loc1 = [loc1[d] - loc[..., d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]

        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)
            wt = prod_n([weights_loc[c[d]][d] for d in range(nb_dims)])
            wt = K.expand_dims(wt, -1)
            interp_vol += wt * vol_val

    else:
        roundloc = tf.cast(tf.round(loc), 'int32')
        max_loc = [d - 1 for d in vol.get_shape().as_list()[:-1]]
        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

    return interp_vol

def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod

def sub2ind(siz, subs, **kwargs):
    k = np.cumprod(siz[::-1])
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]
    return ndx

def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    nb_dims = len(volshape)
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))

    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])

    loc = tf.clip_by_value(loc, 0, [d - 1 for d in volshape])
    return loc - tf.stack(mesh, axis=nb_dims)

def volshape_to_meshgrid(volshape, **kwargs):
    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)

def flatten(v):
    return tf.reshape(v, [-1])

def meshgrid(*args, **kwargs):
    indexing = kwargs.pop("indexing", "xy")

    ndim = len(args)
    s0 = (1,) * ndim

    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))

    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    for i in range(len(output)):       
        output[i] = tf.tile(output[i], tf.stack([*sz[:i], 1, *sz[(i+1):]]))
    return output

def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    volshape = loc_shift.shape[:-1].as_list()
    nb_dims = len(volshape)
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    return interpn(vol, loc, interp_method=interp_method)
