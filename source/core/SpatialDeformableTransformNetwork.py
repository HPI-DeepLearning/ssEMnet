#with identity initalization and deformable ST

from keras.layers.core import Layer
from keras.layers import MaxPooling2D, Conv2D, Dense, Activation, Flatten, Input
from keras.models import Sequential
import tensorflow as tf
import numpy as np

from .bicubic_interp import bicubic_interp_2d
import config

# taken from https://github.com/HPI-DeepLearning/DIRNet/blob/master/DIRNet-mxnet/convnet.py#L209
def identity_matrix_init(shape, dtype=None):
    return np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten()

def locNet(input_shape):
    '''
    locnet = Sequential()
    locnet.add(Conv2D(6, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
    locnet.add(Conv2D(128, (3, 3), kernel_initializer='he_normal'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    #locnet.add(Conv2D(6, (3, 3), kernel_initializer='he_normal'))
    locnet.add(Conv2D(2, (3, 3), kernel_initializer='he_normal'))
    '''
    locnet = Sequential()
    #locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
    locnet.add(Conv2D(8, (3, 3), kernel_initializer='he_normal', input_shape=input_shape))
    #locnet.add(MaxPooling2D(pool_size=(2, 2)))
    #locnet.add(Conv2D(64, (3, 3), kernel_initializer='he_normal'))
    #locnet.add(MaxPooling2D(pool_size=(2, 2)))
    #locnet.add(Conv2D(8, (3, 3), kernel_initializer='he_normal'))
    #locnet.add(MaxPooling2D(pool_size=(2, 2)))
    #locnet.add(Conv2D(256, (3, 3), kernel_initializer='he_normal'))

    #locnet.add(Flatten())
    #locnet.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    ## locnet.add(Dense(6, kernel_initializer='he_normal'))
    #locnet.add(Dense(6, kernel_initializer='zeros', bias_initializer=identity_matrix_init))  # initalize with ID matrix
    return locnet


class SpatialDeformableTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialDeformableTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights

        # Be sure to call this somewhere!
        # from: https://keras.io/layers/writing-your-own-keras-layers/
        super(SpatialDeformableTransformer, self).build(input_shape)

        # self.regularizers = self.locnet.regularizers //NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        #self.constraints = self.locnet.constraints #TODO: einkommentieren und fixen

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        print("input_shape for comp out: ", input_shape)
        print("output_shape in comp out: ", output_size)
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))


    def call(self, X, mask=None):

        print("call X shape is: ", X.shape)
        trans_params = self.locnet.call(X)
        print("deformable transformation from call", trans_params)

        # X should be num_img, height, width, depth
        #self.x = tf.placeholder(tf.float32, config.img_shape)
        #self.y = tf.placeholder(tf.float32, config.img_shape)
        #self.xy = tf.concat([self.x, self.y], 3)

        #X_new = tf.placeholder(tf.float32, [config.num_samples, config.image_height, config.image_depth, 2])
        print("out size is: ", self.output_size)
        output = self._transform(trans_params, X, self.output_size)
        return output

    def _repeat(self, x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(self, im, x, y, out_size):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = self._repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

    def _meshgrid(self, height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        grid = tf.concat([x_t_flat, y_t_flat], 0)
        return grid

    def _transform(self, V, U, out_size):
        print("transform input", tf.shape(U))
        num_batch = tf.shape(U)[0]
        height = tf.shape(U)[1]
        width = tf.shape(U)[2]
        num_channels = tf.shape(U)[3]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        grid = self._meshgrid(out_height, out_width)  # [2, h*w]
        grid = tf.reshape(grid, [-1])  # [2*h*w]
        grid = tf.tile(grid, tf.stack([num_batch]))  # [n*2*h*w]
        grid = tf.reshape(grid, tf.stack([num_batch, 2, -1]))  # [n, 2, h*w]

        # transform (x, y)^T -> (x+vx, x+vy)^T
        V = bicubic_interp_2d(V, out_size)
        V = tf.transpose(V, [0, 3, 1, 2])  # [n, 2, h, w]
        V = tf.reshape(V, [num_batch, 2, -1])  # [n, 2, h*w]
        T_g = tf.add(V, grid)  # [n, 2, h*w]

        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = self._interpolate(
            U, x_s_flat, y_s_flat, out_size)

        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_height, out_width, num_channels]))
        return output