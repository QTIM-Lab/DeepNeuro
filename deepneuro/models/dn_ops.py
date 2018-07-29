
import tensorflow as tf
from keras.layers import UpSampling3D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization


class DnOp(object):

    def __init__(self):

        return


class batch_norm(object):

    # Taken from DCGAN-tensorflow on Github. In future, rewrite for multi-backend batchnorm.

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
        
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train)


def leaky_relu(x, leak=0.2, backend='tf', name="lrelu"):
    
    if backend == 'tf':
        return tf.maximum(x, leak * x)


def relu(backend='tf'):

    if backend == 'tf':
        return tf.nn.relu


def tanh(backend='tf'):

    if backend == 'tf':
        return tf.nn.tanh


def sigmoid(backend='tf'):
    
    if backend == 'tf':
        return tf.nn.sigmoid


def dense(tensor, output_size, stddev=0.02, bias_start=0.0, with_w=False, backend='tf', scope=False):

    if backend == 'tf':

        with tf.variable_scope(scope or "Linear"):

            shape = tensor.get_shape().as_list()

            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer())

            if with_w:
                return tf.matmul(tensor, matrix) + bias, matrix, bias
            else:
                return tf.matmul(tensor, matrix) + bias


def reshape(backend='tf'):

    if backend == 'tf':
        return tf.reshape

    return 


def DnMaxPooling(input_, output_dim, kernel_size=(5, 5), stride_size=(2, 2), dim=2, padding='SAME', initializer_std=0.02, activation=None, name=None, backend='tf'):

    op = None

    if backend == 'keras':

        if dim == 2:
            pass
        if dim == 3:
            pass

    if backend == 'tensorflow':

        if dim == 2:
            pass
        if dim == 3:
            pass

    if op is None:
        print 'Option Not Implemented'

    return op


def DnAveragePooling(input_, ratio=(2, 2), dim=2, backend='tensorflow'):

    op = None

    if backend == 'keras':

        if dim == 2:
            pass
        if dim == 3:
            pass

    if backend == 'tensorflow':

        if dim == 2:
            op = tf.nn.avg_pool(input_, ksize=[1] + list(ratio) + [1], strides=[1] + list(ratio) + [1], padding='SAME')
        if dim == 3:
            op = tf.nn.avg_pool3d(input_, ksize=[1] + list(ratio) + [1], strides=[1] + list(ratio) + [1], padding='SAME')

    if op is None:
        print 'Operation Not Implemented'

    return op


def DnConv(input_, output_dim, kernel_size=(5, 5, 5), stride_size=(2, 2, 2), dim=3, padding='SAME', initializer_std=0.02, activation=None, name=None, backend='tensorflow'):

    """ TODO: Provide different options for intializers, and resolve inconsistencies bewteen 2D and 3D.
    """

    if name is None:
        name = 'conv' + str(dim) + 'd'

    if backend == 'tensorflow':
        if dim == 2:
            conv = conv2d(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name, backend=backend)
        elif dim == 3:
            conv = conv3d(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name, backend=backend)

    elif backend == 'keras':
        if dim == 2:
            conv = Conv2D(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name, backend=backend)
        elif dim == 3:
            conv = Conv3D(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name, backend=backend)

    return conv


def conv2d(input_, output_dim, kernel_size=(3, 3), stride_size=(2, 2), initializer_std=0.02, padding='SAME', name="conv2d", backend='tf'):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=initializer_std))
        
        if padding == 'Other':
            # Not sure what's up with this r n --andrew
            # Something about going from latent space to first conv.
            padding = 'VALID'
            input_ = tf.pad(input_, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(input_, w, strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv3d(input_, output_dim, kernel_size=(3, 3, 3), stride_size=(2, 2, 2), initializer_std=0.02, name="conv3d", backend='tf', padding='SAME', with_w=False):

    with tf.variable_scope(name):

        w = tf.get_variable('w', list(kernel_size) + [input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())

        if padding == 'Other':
            # Not sure what's up with this r n --andrew
            # Something about going from latent space to first conv.
            padding = 'VALID'
            input_ = tf.pad(input_, [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv3d(input_, w, strides=[1] + list(stride_size) + [1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        if with_w:
            return conv, w, biases
        else:
            return conv

        # w = tf.get_variable('w', [kernel_size[0], kernel_size[1], kernel_size[2], input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        
        # print input_, output_dim
        # conv = tf.layers.conv3d(input_, w, kernel_size=list(kernel_size), strides=[stride_size[0], stride_size[1], stride_size[2]], padding=padding)
        # biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer())
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        # return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.zeros_initializer())
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def deconv3d(input_, output_shape, kernel_size=(5, 5, 5), stride_size=(2, 2, 2), stddev=0.02, name="deconv3d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]

        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], kernel_size[2], output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.contrib.layers.xavier_initializer())

        try:
            deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, stride_size[0], stride_size[1], stride_size[2], 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv3d(input_, w, output_shape=output_shape,
                    strides=[1, stride_size[0], stride_size[1], stride_size[2], 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.zeros_initializer())
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def pixel_norm(input, eps=1e-8):
    return input / tf.sqrt(tf.reduce_mean(input**2, axis=3, keep_dims=True) + eps)


def adjusted_std(x, **kwargs): 
    return tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)


def minibatch_state_concat(_input, averaging='all'):

    # Rewrite this later, and understand it --andrew
    
    vals = adjusted_std(_input, axis=0, keep_dims=True)
    
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print "nothing"

    multiples = tuple([int(_input.shape[0]), 4, 4, 1])
    vals = tf.tile(vals, multiples=multiples)  # Be aware, need updated TF for this to work.
    
    return tf.concat([_input, vals], axis=3)


# Some of the following functions may be redundant


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape


def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x


def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))


def downscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h / scale, w / scale))


def UpConvolution(deconvolution=False, pool_size=(2, 2, 2), implementation='keras'):

    """ Keras doesn't have native support for deconvolution yet, but keras_contrib does.
        If deconvolution is not specified, normal upsampling will be used.

        TODO: Currently only works in 2D.
        TODO: Rewrite in style of other dn_ops

        Parameters
        ----------
        deconvolution : bool, optional
            If true, will attempt to load Deconvolutio from keras_contrib
        pool_size : tuple, optional
            Upsampling ratio along each axis.
        implementation : str, optional
            Specify 'keras' or 'tensorflow' implementation.
        
        Returns
        -------
        Keras Tensor Operation
            Either Upsampling3D() or Deconvolution()
    """

    if implementation == 'keras':
        if not deconvolution:
            return UpSampling3D(size=pool_size)
        else:
            return None
            # deconvolution not yet implemented // required from keras_contrib
