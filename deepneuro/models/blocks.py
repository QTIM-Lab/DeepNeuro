import numpy as np

from deepneuro.models.dn_ops import DnConv, DnPixelNorm, DnUpsampling, DnMaxPooling, DnBatchNormalization, DnDropout, DnAveragePooling
from deepneuro.models.ops import leaky_relu, minibatch_state_concat


def generator(model, latent_var, depth=1, initial_size=(4, 4), max_size=None, reuse=False, transition=False, alpha_transition=0, name=None):
    
    """Summary
    
    Parameters
    ----------
    model : TYPE
        Description
    latent_var : TYPE
        Description
    depth : int, optional
        Description
    initial_size : tuple, optional
        Description
    max_size : None, optional
        Description
    reuse : bool, optional
        Description
    transition : bool, optional
        Description
    alpha_transition : int, optional
        Description
    name : None, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    
    import tensorflow as tf

    with tf.variable_scope(name) as scope:

        convs = []

        if reuse:
            scope.reuse_variables()

        convs += [tf.reshape(latent_var, [tf.shape(latent_var)[0]] + [1] * model.dim + [model.latent_size])]

        # TODO: refactor the padding on this step. Or replace with a dense layer?
        with tf.variable_scope('generator_n_conv_1_{}'.format(convs[-1].shape[1])):
            convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(0), kernel_size=initial_size, stride_size=(1,) * model.dim, padding='Other', dim=model.dim)), model.dim)

        convs += [tf.reshape(convs[-1], [tf.shape(latent_var)[0]] + list(initial_size) + [model.get_filter_num(0)])]

        with tf.variable_scope('generator_n_conv_2_{}'.format(convs[-1].shape[1])):
            convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(0), kernel_size=(5,) * model.dim, stride_size=(1,) * model.dim, dim=model.dim)), dim=model.dim)

        for i in range(depth):

            # Calculate Next Upsample Ratio
            if max_size is None:
                upsample_ratio = (2,) * model.dim
            else:
                upsample_ratio = []
                for size_idx, size in enumerate(max_size):
                    if size >= convs[-1].shape[size_idx + 1] * 2:
                        upsample_ratio += [2]
                    else:
                        upsample_ratio += [1]
                upsample_ratio = tuple(upsample_ratio)

            # Upsampling, with conversion to RGB if necessary.
            if i == depth - 1 and transition:
                transition_conv = DnConv(convs[-1], output_dim=model.channels, kernel_size=(1,) * model.dim, stride_size=(1,) * model.dim, dim=model.dim, name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]))
                transition_conv = DnUpsampling(transition_conv, upsample_ratio, dim=model.dim)

            convs += [DnUpsampling(convs[-1], upsample_ratio, dim=model.dim)]

            # Convolutional blocks. TODO: Replace with block module.
            with tf.variable_scope('generator_n_conv_1_{}'.format(convs[-1].shape[1])):
                convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(i + 1), kernel_size=(5,) * model.dim, stride_size=(1,) * model.dim, dim=model.dim)), dim=model.dim)
            with tf.variable_scope('generator_n_conv_2_{}'.format(convs[-1].shape[1])):
                convs += [DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(i + 1), kernel_size=(5,) * model.dim, stride_size=(1,) * model.dim, dim=model.dim)), dim=model.dim)]

        # Conversion to RGB
        convs += [DnConv(convs[-1], output_dim=model.channels, kernel_size=(1,) * model.dim, stride_size=(1,) * model.dim, name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=model.dim)]

        if transition:
            convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

        return convs[-1]


def discriminator(model, input_image, reuse=False, initial_size=(4, 4), max_size=None, name=None, depth=1, transition=False, alpha_transition=0, **kwargs):

    import tensorflow as tf

    """
    """

    with tf.variable_scope(name) as scope:

        convs = []

        if reuse:
            scope.reuse_variables()

        # fromRGB
        convs += [leaky_relu(DnConv(input_image, output_dim=model.get_filter_num(depth), kernel_size=(1,) * model.dim, stride_size=(1,) * model.dim, name='discriminator_y_rgb_conv_{}'.format(input_image.shape[1]), dim=model.dim))]

        for i in range(depth):

            # Convolutional blocks. TODO: Replace with block module.
            convs += [leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(depth - i), kernel_size=(5,) * model.dim, stride_size=(1,) * model.dim, name='discriminator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=model.dim))]
            convs += [leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(depth - 1 - i), kernel_size=(5,) * model.dim, stride_size=(1,) * model.dim, name='discriminator_n_conv_2_{}'.format(convs[-1].shape[1]), dim=model.dim))]
            
            # Calculate Next Downsample Ratio
            # Whoever can calculate this in a less dumb way than this gets a Fields Medal.
            if max_size is None:
                downsample_ratio = (2,) * model.dim
            else:
                reference_shape = []
                current_shape = input_image.shape
                for idx, cshape in enumerate(current_shape):
                    reference_shape += [current_shape[idx] // initial_size[idx]]
                downsample_ratio = []
                for size_idx, size in enumerate(max_size):
                    if size // initial_size[size_idx] > min(reference_shape):
                        downsample_ratio += [1]
                    else:
                        downsample_ratio += [2]
                downsample_ratio = tuple(downsample_ratio)

            convs[-1] = DnAveragePooling(convs[-1], downsample_ratio, dim=model.dim)

            if i == 0 and transition:
                transition_conv = DnAveragePooling(input_image, downsample_ratio, dim=model.dim)
                transition_conv = leaky_relu(DnConv(transition_conv, output_dim=model.get_filter_num(depth - 1), kernel_size=(1,) * model.dim, stride_size=(1,) * model.dim, name='discriminator_y_rgb_conv_{}'.format(transition_conv.shape[1]), dim=model.dim))
                convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

        convs += [minibatch_state_concat(convs[-1])]
        convs[-1] = leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(0), kernel_size=(3,) * model.dim, stride_size=(1,) * model.dim, name='discriminator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=model.dim))

        output = tf.reshape(convs[-1], [tf.shape(convs[-1])[0], np.prod(initial_size) * model.get_filter_num(0)])

        # Currently erroring
        # discriminate_output = dense(output, output_size=1, name='discriminator_n_fully')

        discriminate_output = tf.layers.dense(output, 1, name='discriminator_n_1_fully')

        return tf.nn.sigmoid(discriminate_output), discriminate_output


def unet(model, input_tensor, backend='tensorflow'):

        from keras.layers.merge import concatenate

        left_outputs = []

        for level in range(model.depth):

            filter_num = int(model.max_filter / (2 ** (model.depth - level)) / model.downsize_filters_factor)

            if level == 0:
                left_outputs += [DnConv(input_tensor, filter_num, model.kernel_size, stride_size=(1,) * model.dim, activation=model.activation, padding=model.padding, dim=model.dim, name='unet_downsampling_conv_{}_1'.format(level), backend=backend)]
                left_outputs[level] = DnConv(left_outputs[level], 2 * filter_num, model.kernel_size, stride_size=(1,) * model.dim, activation=model.activation, padding=model.padding, dim=model.dim, name='unet_downsampling_conv_{}_2'.format(level), backend=backend)
            else:
                left_outputs += [DnMaxPooling(left_outputs[level - 1], pool_size=model.pool_size, dim=model.dim, backend=backend)]
                left_outputs[level] = DnConv(left_outputs[level], filter_num, model.kernel_size, stride_size=(1,) * model.dim, activation=model.activation, padding=model.padding, dim=model.dim, name='unet_downsampling_conv_{}_1'.format(level), backend=backend)
                left_outputs[level] = DnConv(left_outputs[level], 2 * filter_num, model.kernel_size, stride_size=(1,) * model.dim, activation=model.activation, padding=model.padding, dim=model.dim, name='unet_downsampling_conv_{}_2'.format(level), backend=backend)

            if model.dropout is not None and model.dropout != 0:
                left_outputs[level] = DnDropout(model.dropout)(left_outputs[level])

            if model.batch_norm:
                left_outputs[level] = DnBatchNormalization(left_outputs[level])

        right_outputs = [left_outputs[model.depth - 1]]

        for level in range(model.depth):

            filter_num = int(model.max_filter / (2 ** (level)) / model.downsize_filters_factor)

            if level > 0:
                right_outputs += [DnUpsampling(right_outputs[level - 1], pool_size=model.pool_size, dim=model.dim, backend=backend)]
                right_outputs[level] = concatenate([right_outputs[level], left_outputs[model.depth - level - 1]], axis=model.dim + 1)
                right_outputs[level] = DnConv(right_outputs[level], filter_num, model.kernel_size, stride_size=(1,) * model.dim, activation=model.activation, padding=model.padding, dim=model.dim, name='unet_upsampling_conv_{}_1'.format(level), backend=backend)
                right_outputs[level] = DnConv(right_outputs[level], int(filter_num / 2), model.kernel_size, stride_size=(1,) * model.dim, activation=model.activation, padding=model.padding, dim=model.dim, name='unet_upsampling_conv_{}_2'.format(level), backend=backend)
            else:
                continue

            if model.dropout is not None and model.dropout != 0:
                right_outputs[level] = DnDropout(model.dropout)(right_outputs[level])

            if model.batch_norm:
                right_outputs[level] = DnBatchNormalization()(right_outputs[level])

        output_layer = DnConv(right_outputs[level], 1, (1, ) * model.dim, stride_size=(1,) * model.dim, dim=model.dim, name='end_conv', backend=backend) 

        # TODO: Brainstorm better way to specify outputs
        if model.input_tensor is not None:
            return output_layer

        return model.model