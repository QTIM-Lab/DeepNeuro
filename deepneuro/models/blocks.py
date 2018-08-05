import tensorflow as tf

from deepneuro.models.dn_ops import DnConv, DnPixelNorm, DnUpsampling
from deepneuro.models.ops import leaky_relu


def generator(model, latent_var, depth=1, initial_size=4, reuse=False, name=None):

    convs = []

    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables()

        convs += [tf.reshape(latent_var, [model.training_batch_size] + [1] * model.dim + [model.latent_size])]

        # TODO: refactor the padding on this step.
        convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(0), kernel_size=(4,) * model.dim, stride_size=(1,) * model.dim, padding='Other', name='generator_conv_1_latent', dim=model.dim)), model.dim)

        convs += [tf.reshape(convs[-1], [model.training_batch_size] + [initial_size] * model.dim + [model.get_filter_num(0)])]

        convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(0), stride_size=(1,) * model.dim, name='generator_conv_2_latent', dim=model.dim)), dim=model.dim)

        for i in range(depth):

            convs += [DnUpsampling(convs[-1], (2,) * model.dim, dim=model.dim)]
            convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(i + 1), stride_size=(1,) * model.dim, name='generator_conv_1_depth_{}_{}'.format(i, convs[-1].shape[1]), dim=model.dim)), dim=model.dim)

            convs += [DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(i + 1), stride_size=(1,) * model.dim, name='generator_conv_2_depth_{}_{}'.format(i, convs[-1].shape[1]), dim=model.dim)), dim=model.dim)]

        #To RGB
        convs += [DnConv(convs[-1], output_dim=model.channels, kernel_size=(1,) * model.dim, stride_size=(1,) * model.dim, name='generator_y_final_conv_{}'.format(convs[-1].shape[1]), dim=model.dim)]

        return convs[-1]


# def progressive_generator(model, latent_var, progressive_depth=1, name=None, transition=False, alpha_transition=0.0):

#     with tf.variable_scope(name) as scope:

#         convs = []

#         convs += [tf.reshape(latent_var, [model.training_batch_size, 1, 1, model.latent_size])]

#         convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(1, depth), kernel_size=(4, 4), stride_size=(1,) * model.dim, padding='Other', name='generator_n_1_conv', dim=model.dim)))

#         convs += [tf.reshape(convs[-1], [model.training_batch_size, 4, 4, model.get_filter_num(1, depth)])] # why necessary? --andrew
#         convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(1, depth), stride_size=(1,) * model.dim, name='generator_n_2_conv', dim=model.dim)))

#         for i in range(progressive_depth - 1):

#             if i == progressive_depth - 2 and transition:  # redundant conditions? --andrew
#                 #To RGB
#                 # Don't totally understand this yet, diagram out --andrew
#                 transition_conv = DnConv(convs[-1], output_dim=model.channels, kernel_size=(1, 1), stride_size=(1,) * model.dim, name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=model.dim)
#                 transition_conv = upscale(transition_conv, 2)

#             convs += [upscale(convs[-1], 2)]
#             convs[-1] = DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(i + 1, depth), stride_size=(1,) * model.dim, name='generator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=model.dim)))

#             convs += [DnPixelNorm(leaky_relu(DnConv(convs[-1], output_dim=model.get_filter_num(i + 1, depth), stride_size=(1,) * model.dim, name='generator_n_conv_2_{}'.format(convs[-1].shape[1]), dim=model.dim)))]

#         #To RGB
#         convs += [DnConv(convs[-1], output_dim=model.channels, kernel_size=(1, 1), stride_size=(1,) * model.dim, name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=model.dim)]

#         if transition:
#             convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

#         return convs[-1]