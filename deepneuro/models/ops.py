""" Basic TensorFlow operations for use in DeepNeuroOps and DeepNeuro Blocks.
"""

import tensorflow as tf

# def upscale2d_conv2d(input_tensor, output_dim, scale=2, kernel_size=(3, 3), stride_size=(2, 2), initializer_std=0.02, gain=np.sqrt(2), use_wscale=False, name="upscale_conv2d"):

#     """ Reimplemented from https://github.com/NVlabs/stylegan.
#     Faster and uses less memory than performing the operations separately.
#     """

#     w = tf.get_variable('w', [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=initializer_std))

#     w = tf.pad(w, [[1, 1], [0, 0], [0, 0], [1, 1]], mode='CONSTANT')
#     w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
#     w = tf.cast(w, input_tensor.dtype)

#     os = [tf.shape(x)[0], output_dim, x.shape[2] * 2, x.shape[3] * 2]
#     return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')


def conv2d(input_tensor, output_dim, kernel_size=(3, 3), stride_size=(2, 2), initializer_std=0.02, padding='SAME', name="conv2d"):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], input_tensor.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=initializer_std))
        
        if padding == 'OTHER':
            padding = 'VALID'
            input_tensor = tf.pad(input_tensor, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(input_tensor, w, strides=[1, stride_size[0], stride_size[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def conv3d(input_tensor, output_dim, kernel_size=(3, 3, 3), stride_size=(2, 2, 2), initializer_std=0.02, name="conv3d", backend='tf', padding='SAME', with_w=False):

    with tf.variable_scope(name):

        w = tf.get_variable('w', list(kernel_size) + [input_tensor.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())

        if padding == 'Other':
            # GAN-Specific Setting. Consider removal.
            padding = 'VALID'
            input_tensor = tf.pad(input_tensor, [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv3d(input_tensor, w, strides=[1] + list(stride_size) + [1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        if with_w:
            return conv, w, biases
        else:
            return conv


def deconv2d(input_tensor, output_dim, kernel_size=(3, 3), stride_size=(2, 2), stddev=0.02, name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], output_dim[-1], input_tensor.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_tensor, w, output_shape=output_dim,
                    strides=[1, stride_size[0], stride_size[1], 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_tensor, w, output_shape=output_dim,
                    strides=[1, stride_size[0], stride_size[1], 1])

        biases = tf.get_variable('biases', [output_dim[-1]], initializer=tf.zeros_initializer())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def deconv3d(input_tensor, output_shape, kernel_size=(5, 5, 5), stride_size=(2, 2, 2), stddev=0.02, name="deconv3d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]

        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], kernel_size[2], output_shape[-1], input_tensor.get_shape()[-1]],
                  initializer=tf.contrib.layers.xavier_initializer())

        try:
            deconv = tf.nn.conv3d_transpose(input_tensor, w, output_shape=output_shape, strides=[1, stride_size[0], stride_size[1], stride_size[2], 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv3d(input_tensor, w, output_shape=output_shape,
                    strides=[1, stride_size[0], stride_size[1], stride_size[2], 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.zeros_initializer())
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def pixel_norm_2d(input_tensor, eps=1e-8):
    return input_tensor / tf.sqrt(tf.reduce_mean(input_tensor ** 2, axis=3, keepdims=True) + eps)


def pixel_norm_3d(input_tensor, eps=1e-8):
    return input_tensor / tf.sqrt(tf.reduce_mean(input_tensor ** 2, axis=4, keepdims=True) + eps)


def adjusted_std(x, **kwargs): 
    return tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)


def minibatch_state_concat(input_tensor, averaging='all', dim=2):
    
    vals = adjusted_std(input_tensor, axis=0, keepdims=True)
    
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keepdims=True)
    else:
        raise NotImplementedError

    batch_size = tf.shape(input_tensor)[0]
    # A little weird, because + is overloaded by Tensorflow
    multiples = [1] * (dim + 1)
    multiples[0] = batch_size
    multiples[1:-1] = input_tensor.shape[1:-1]
    multiples = (4,) * dim + (1,)
    multiples = (batch_size,) + multiples

    vals = tf.tile(vals, multiples=multiples)  # Be aware, need updated TF for this to work.
    
    return tf.concat([input_tensor, vals], axis=dim + 1)


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


def upscale2d(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))


def downscale2d(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (int(h / scale), int(w / scale)))


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


def dense(tensor, output_size, stddev=0.02, bias_start=0.0, with_w=False, backend='tensorflow', name="dense"):

    if backend == 'tensorflow':

        with tf.variable_scope(name):

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