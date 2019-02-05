
import tensorflow as tf

from keras.layers import UpSampling2D, UpSampling3D, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, ZeroPadding2D, ZeroPadding3D

from deepneuro.models.ops import pixel_norm_2d, pixel_norm_3d, conv2d, conv3d, deconv2d, deconv3d, upscale2d


class DnOp(object):

    def __init__(self):

        return


def DnDropout(input_, ratio=.5, backend='tensorflow'):

    if backend == 'keras':

        return Dropout(ratio)(input_)

    if backend == 'tensorflow':

        return tf.nn.dropout(input_, ratio)

    return


def DnBatchNormalization(input_, backend='tensorflow'):

    if backend == 'keras' or True:

        return BatchNormalization()(input_)

    if backend == 'tensorflow':

        return tf.contrib.layers.batch_norm(input_)

    return


def DnMaxPooling(input_, pool_size=(2, 2), dim=2, padding='SAME', backend='tensorflow'):

    op = None

    if backend == 'keras':

        if dim == 2:
            return MaxPooling2D(pool_size=pool_size, padding=padding)(input_)
        if dim == 3:
            return MaxPooling3D(pool_size=pool_size, padding=padding)(input_)

    if backend == 'tensorflow':

        if dim == 2:
            op = tf.nn.max_pool(input_, ksize=[1] + list(pool_size) + [1], strides=[1] + list(pool_size) + [1], padding='SAME')
        if dim == 3:
            op = tf.nn.max_pool3d(input_, ksize=[1] + list(pool_size) + [1], strides=[1] + list(pool_size) + [1], padding='SAME')

    if op is None:
        raise NotImplementedError

    return op


def DnAveragePooling(input_, pool_size=(2, 2), dim=2, backend='tensorflow'):

    if backend == 'keras':

        if dim == 2:
            raise NotImplementedError
        if dim == 3:
            raise NotImplementedError

    if backend == 'tensorflow':

        if dim == 2:
            op = tf.nn.avg_pool(input_, ksize=[1] + list(pool_size) + [1], strides=[1] + list(pool_size) + [1], padding='SAME')
        if dim == 3:
            op = tf.nn.avg_pool3d(input_, ksize=[1] + list(pool_size) + [1], strides=[1] + list(pool_size) + [1], padding='SAME')

    return op


def DnConv(input_, output_dim, kernel_size=(5, 5, 5), stride_size=(2, 2, 2), dim=3, padding='SAME', initializer_std=0.02, activation=None, name=None, backend='tensorflow'):

    """ TODO: Provide different options for intializers, and resolve inconsistencies bewteen 2D and 3D.
    """

    if name is None:
        name = 'conv' + str(dim) + 'd'

    if backend == 'tensorflow':

        padding = padding.upper()

        if dim == 2:
            conv = conv2d(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name)
        elif dim == 3:
            conv = conv3d(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name)

    elif backend == 'keras':
        if dim == 2:
            conv = Conv2D(output_dim, kernel_size=kernel_size, strides=stride_size, padding=padding, name=name)(input_)
        elif dim == 3:
            conv = Conv3D(output_dim, kernel_size=kernel_size, strides=stride_size, padding=padding, name=name)(input_)

    if activation is not None:
        conv = Activation('sigmoid')(conv)

    return conv


def DnDeConv(input_, output_dim, kernel_size=(5, 5, 5), stride_size=(2, 2, 2), dim=3, padding='SAME', initializer_std=0.02, activation=None, name=None, backend='tensorflow'):

    """ TODO: Provide different options for intializers, and resolve inconsistencies bewteen 2D and 3D.
    """

    if name is None:
        name = 'conv' + str(dim) + 'd'

    if backend == 'tensorflow':

        padding = padding.upper()

        if dim == 2:
            conv = deconv2d(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name, backend=backend)
        elif dim == 3:
            conv = deconv3d(input_, output_dim, kernel_size=kernel_size, stride_size=stride_size, padding=padding, initializer_std=initializer_std, name=name, backend=backend)

    elif backend == 'keras':
        
        print('not implemented')

    if activation is not None:
        pass

    return conv


def DnUpsampling(input_, pool_size=(2, 2, 2), dim=3, backend='tensorflow'):

    if backend == 'keras':
        if dim == 2:
            return UpSampling2D(size=pool_size)(input_)
        elif dim == 3:
            return UpSampling3D(size=pool_size)(input_)

    if backend == 'tensorflow':
        if dim == 2:
            return upscale2d(input_, pool_size[0])
        elif dim == 3:
            raise NotImplementedError


def DnPixelNorm(input_, dim=3, backend='tensorflow'):

    if backend == 'tensorflow':

        if dim == 2:
            return pixel_norm_2d(input_)
        elif dim == 3:
            return pixel_norm_3d(input_)

    if backend == 'keras':
        raise NotImplementedError


def DnZeroPadding(input_, padding=(1, 1), dim=3, backend='keras'):

    if backend == 'keras':

        if dim == 2:
            return ZeroPadding2D(padding=padding)(input_)
        elif dim == 3:
            print(padding)
            return ZeroPadding3D(padding=padding)(input_)

    if backend == 'tensorflow':
        raise NotImplementedError

