# From https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm, variance_scaling_initializer

# the implements of leakyRelu
def lrelu(x , alpha= 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', name="conv2d", with_w=False):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=variance_scaling_initializer())

        if padding == 'Other':
            # Not sure what's up with this r n --andrew
            # Something about going from latent space to first conv.
            padding = 'VALID'
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases
        else:
            return conv

def conv3d(input_, output_dim, k_h=3, k_w=3, k_d=3, d_h=2, d_w=2, d_d=3, padding='SAME', name="conv3d", with_w=False):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, k_d, input_.get_shape()[-1], output_dim], initializer=variance_scaling_initializer())

        if padding == 'Other':
            # Not sure what's up with this r n --andrew
            # Something about going from latent space to first conv.
            padding = 'VALID'
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_d, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        if with_w:
            return conv, w, biases
        else:
            return conv


def de_conv(input_, output_shape, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=variance_scaling_initializer())

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def de_conv_3d(input_, output_shape, k_h=3, k_w=3, k_d=3, d_h=2, d_w=2, d_d=2, stddev=0.02, name="deconv3d", with_w=False):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, k_d, output_shape[-1], input_.get_shape()[-1]], initializer=variance_scaling_initializer())

        try:
            deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, d_d, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            # No support in this case..
            raise
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, d_d, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def fully_connect(input_, output_size, stddev=0.02, scope=None, with_w=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):

        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, variance_scaling_initializer())

        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))

        output = tf.matmul(input_, matrix) + bias

        if with_w:
            return output, with_w, bias

        else:
            return output


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""

    # Unfinished -- but not needed??
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(4, [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])


def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse= reuse , updates_collections=None)


def resize_nearest_neighbor(x, new_size, dims):
    
    # Weird discussion here about the best way to do this:
    # https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images

    # 2D Built-in Method
    # x = tf.image.resize_nearest_neighbor(x, new_size)

    # Weirdo method by applying 2D twice.

    axes = [1,2]
    sizes = [[new_size[0], new_size[2]], [new_size[0], new_size[1]]]
    for size, axis in zip(sizes,axes):
        resized_list = []
        unstack_img_depth_list = tf.unstack(x, axis=axis)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, size, method=0))
        x = tf.stack(resized_list, axis=axis)

    return x


def upscale(x, scale):

    # For 2d...
    b, h, w, d, n = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale, d * scale), [b, h, w, d, n])

    # Upscaling-only solution..
    # https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images

    # isolate = tf.transpose(x,[0,4,1,2,3])  # [batch_size,n,width,height,depth]
    # flatten_it_all = tf.reshape([b * n * w * h * d, 1])  # flatten it
    # expanded_it = flatten_it_all * tf.ones([1, scale ** 3])
    # prepare_for_transpose = tf.reshape(expanded_it, [b * n, w, h, d, scale, scale, scale])

    # # This is magical to me, study it more.
    # transpose_to_align_neighbors = tf.transpose(prepare_for_transpose, [0,1,6,2,5,3,4])
    # expand_it_all = tf.reshape(transpose_to_align_neighbors, [b, n, w*scale, h*scale, d*scale])

    # # then finally reorder and you are done
    # reorder_dimensions = tf.transpose(expand_it_all,[0,2,3,4,1])  # [batch_size,width*2,height*2,depth*2,n]

    # return reorder_dimensions


def downscale(x, scale):

    b, h, w, d, n = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h / scale, w / scale, d / scale), [b, h, w, d, n])


def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def avgpool3d(x, k=2):
    return tf.nn.avg_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1], padding='SAME')


def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[4]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2,3], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv

        return scale*normalized + offset


def pixel_norm(input, eps=1e-8):
    return input / tf.sqrt(tf.reduce_mean(input**2, axis=4, keep_dims=True) + eps)


def minibatch_state_concat(input, averaging='all'):

    # Rewrite this later --andrew

    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)

    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print "nothing"

    multiples = tuple([int(input.shape[0]), 4, 4, 4, 1])
    vals = tf.tile(vals, multiples=multiples) # Be aware, need updated TF for this to work.
    return tf.concat([input, vals], axis=4)


class WScaleLayer(object):

    def __int__(self, weights, biases):

        self.scale = tf.sqrt(tf.reduce_mean(weights ** 2))
        self.bias = None
        self.we_assign = weights.assign(weights / self.scale)
        if biases is not None:
            self.bias = biases

    def getoutput_for(self, input):

        # Something is clearly messed up here --andrew
        if self.bias is not None:
            input = input - self.bias

        return input * self.scale + self.bias, self.we_assign