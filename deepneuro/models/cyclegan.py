""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

import tensorflow as tf

from keras.engine import Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Dropout, BatchNormalization, Flatten, Dense
from keras.optimizers import Nadam
from keras.layers.merge import concatenate

from deepneuro.models.model import TensorFlowModel
from deepneuro.models.cost_functions import dice_coef_loss, dice_coef
from deepneuro.models.dn_ops import UpConvolution, DnConv
from deepneuro.utilities.util import add_parameter
from ops_three import lrelu, conv3d, fully_connect, downscale, upscale, pixel_norm, avgpool3d, WScaleLayer, minibatch_state_concat


class CycleGan(TensorFlowModel):
    
    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        super(CycleGan, self).load(kwargs)

        add_parameter(self, kwargs, 'dim', 3)

        # Generator Parameters
        add_parameter(self, kwargs, 'generator_depth', 4)
        add_parameter(self, kwargs, 'generator_max_filter', 64)

        # Discriminator Parameters
        add_parameter(self, kwargs, 'discriminator_depth', 4)
        add_parameter(self, kwargs, 'discriminator_max_filter', 64)

        # Training Parameters
        add_parameter(self, kwargs, 'consistency_weight', 100)  # AKA lambda
        add_parameter(self, kwargs, 'gradient_penalty_weight', 10)

        self.sess = None
        self.init = None

    def train(self, training_data_collection, validation_data_collection=None, **kwargs):

        add_parameter(self, kwargs, 'num_epochs', 100)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 10)

        add_parameter(self, kwargs, 'training_batch_size', 16)

        # training_parameters = {'input_groups': ['input_modalities', 'ground_truth'],
        #                 'output_model_filepath': model_file,
        #                 'training_batch_size': 20,
        #                 'validation_batch_size': 20,
        #                 'num_epochs': 100,
        #                 'training_steps_per_epoch': 10,
        #                 'validation_steps_per_epoch': 27,
        #                 'save_best_only': True}

        self.build_tensorflow_model(self.training_batch_size)

        self.create_data_generators(training_data_collection, validation_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)

        self.init_sess()

        step = 0
        batch_num = 0

        self.num_epochs = 20
        self.training_steps_per_epoch = 20

        for epoch in range(self.num_epochs):

            for step in range(self.training_steps_per_epoch):
                
                print epoch, step
                input_modality_1, input_modality_2 = next(self.training_data_generator)

                # synth_1_2 = sess.run(self.generator_1_2, feed_dict={self.generator_input_images_1: input_modality_1})
                # synth_2_1 = sess.run(self.generator_2_1, feed_dict={self.generator_input_images_2: input_modality_2})

                # print(synth_1_2.shape)

                # Optimize!
                _, _, discrim_1_loss, discrim_2_loss, d_loss, gen_1_loss, gen_2_loss, cons_1_loss, cons_2_loss, g_loss = self.sess.run([self.generator_optmizer, self.discriminator_optimizer, self.D_loss_wgan_1_2, self.D_loss_wgan_2_1, self.total_D_loss, self.G_loss_1_2, self.G_loss_2_1, self.generator_1_2_loss, self.generator_2_1_loss, self.total_G_loss], feed_dict={self.generator_input_images_1: input_modality_1, self.generator_input_images_2: input_modality_2})

                print('Discrim 1 Loss', discrim_1_loss)
                print('Discrim 2 Loss', discrim_2_loss)
                print('Total D Loss', d_loss)
                print('Gen 1 Loss', gen_1_loss)
                print('Gen 2 Loss', gen_2_loss)
                print('Consitency Loss 12', cons_1_loss)
                print('Consitency Loss 21', cons_2_loss)  
                print('Total G Loss', g_loss)

                # _, cons_1_loss, cons_2_loss, g_loss = self.sess.run([self.consistency_optimizer, self.generator_1_2_loss, self.generator_2_1_loss, self.total_generator_loss], feed_dict={self.generator_input_images_1: input_modality_1, self.generator_input_images_2: input_modality_2})

                # print('Consitency Loss 12', cons_1_loss)
                # print('Consitency Loss 21', cons_2_loss)  
                # print('Total G Loss', g_loss)

        return

    def build_tensorflow_model(self, batch_size):

        # Set input/output shapes for reference during inference.
        self.model_input_shape = tuple([batch_size] + list(self.input_shape))
        self.model_output_shape = tuple([batch_size] + list(self.input_shape))

        # Create input placeholders
        self.generator_input_images_1 = tf.placeholder(tf.float32, [None] + list(self.input_shape))
        self.generator_input_images_2 = tf.placeholder(tf.float32, [None] + list(self.input_shape))

        # Create generators
        self.generator_1_2 = self.generator(self.generator_input_images_1, name='generator_1')
        self.generator_2_1 = self.generator(self.generator_1_2, name='generator_2')

        # Create Consistency Losses
        self.generator_1_2_loss =  tf.reduce_mean((self.generator_1_2 - self.generator_input_images_2) ** 2)
        self.generator_2_1_loss =  tf.reduce_mean((self.generator_2_1 - self.generator_input_images_1) ** 2)

        # Create Discriminators
        self.discriminator_1_2_fake, self.discriminator_1_2_fake_logits = self.discriminator(self.generator_1_2, name='discriminator_1_2')
        self.discriminator_1_2_real, self.discriminator_1_2_real_logits = self.discriminator(self.generator_input_images_2, reuse=True, name='discriminator_1_2')
        self.total_discriminator_1_2_loss = self.discriminator_1_2_real + self.discriminator_1_2_fake

        self.discriminator_2_1_fake, self.discriminator_2_1_fake_logits = self.discriminator(self.generator_2_1, name='discriminator_2_1')
        self.discriminator_2_1_real, self.discriminator_2_1_real_logits = self.discriminator(self.generator_input_images_1, reuse=True, name='discriminator_2_1')
        self.total_discriminator_2_1_loss = self.discriminator_2_1_real + self.discriminator_2_1_fake

        # Create Basic GAN Loss
        self.D_loss_1_2 = tf.reduce_mean(self.discriminator_1_2_fake_logits) - tf.reduce_mean(self.discriminator_1_2_real_logits)
        self.G_loss_1_2 = -tf.reduce_mean(self.discriminator_1_2_fake_logits)
        self.D_loss_2_1 = tf.reduce_mean(self.discriminator_2_1_fake_logits) - tf.reduce_mean(self.discriminator_2_1_real_logits)
        self.G_loss_2_1 = -tf.reduce_mean(self.discriminator_2_1_fake_logits)

        # Wasserstein GANs
        self.D_loss_wgan_1_2 = self.wasserstein_loss(self.D_loss_1_2, self.generator_input_images_2, self.generator_1_2, batch_size, name='discriminator_1_2')
        self.D_loss_wgan_2_1 = self.wasserstein_loss(self.D_loss_2_1, self.generator_input_images_1, self.generator_2_1, batch_size, name='discriminator_2_1')

        self.total_D_loss = self.D_loss_wgan_1_2 + self.D_loss_wgan_2_1
        self.total_generator_loss = self.generator_1_2_loss + self.generator_2_1_loss
        self.total_G_loss = self.total_generator_loss * self.consistency_weight + self.G_loss_1_2 + self.G_loss_2_1

        self.g_vars = [var for var in tf.trainable_variables() if 'gen' in var.name]
        self.d_vars = [var for var in tf.trainable_variables() if 'dis' in var.name]

        self.generator_optmizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.total_G_loss, var_list=self.g_vars)
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.total_D_loss, var_list=self.d_vars)
        self.consistency_optimizer = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.total_generator_loss, var_list=self.g_vars)

        # self.disriminator_1_2_fake = self.discriminator(self.input_images_1)
        # self.disriminator_1_2_real = self.discriminator(self.input_images_1)

    def wasserstein_loss(self, discriminator_loss, real_data, fake_data, batch_size, name=''):

        # Implementation fo Wasserstein loss with gradient penalty.

        # Gradient Penalty from Wasserstein GAN GP, I believe? Check on it --andrew
        # Also investigate more of what's happening here --andrew
        differences = fake_data - real_data
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1, 1], minval=0., maxval=1.)
        interpolates = real_data + (alpha * differences)
        _, discri_logits = self.discriminator(interpolates, name=name, reuse=True)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # Some sort of norm from papers, check up on it. --andrew
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        # Update Loss functions..
        discriminator_loss += self.gradient_penalty_weight * gradient_penalty
        # discriminator_loss += 0.001 * tf.reduce_mean(tf.square(self.discriminator_1_2_real_logits - 0.0))

        return discriminator_loss

    def discriminator(self, input_image, reuse=False, name=None, scope=None, transition=False, alpha_transition=0.01):

        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            convs = []

            # fromRGB
            convs += [lrelu(conv3d(input_image, output_dim=self.discriminator_max_filter / self.discriminator_depth, k_w=1, k_h=1, k_d=1, d_w=1, d_h=1, d_d=1, name='dis_y_rgb_conv_{}'.format(input_image.shape[1])))]

            for i in range(self.discriminator_depth - 1):

                convs += [lrelu(conv3d(convs[-1], output_dim=self.discriminator_max_filter / (self.discriminator_depth - i - 1), d_h=1, d_w=1, d_d=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))]

                convs += [lrelu(conv3d(convs[-1], output_dim=self.discriminator_max_filter / (self.discriminator_depth - 1 - i), d_h=1, d_w=1, d_d=1, name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))]
                convs[-1] = avgpool3d(convs[-1], 2)

            # convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = lrelu(conv3d(convs[-1], output_dim=self.discriminator_max_filter, k_w=3, k_h=3, k_d=3, d_h=1, d_w=1, d_d=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            conv = lrelu(conv3d(convs[-1], output_dim=self.discriminator_max_filter, k_w=4, k_h=4, k_d=4, d_h=1, d_w=1, d_d=1, padding='VALID', name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))
            
            #for D
            output = tf.layers.Flatten()(convs[-1])
            output = fully_connect(output, output_size=1, scope='dis_n_fully')
            # fd = dg

            return tf.nn.sigmoid(output), output

    def generator(self, inputs, num_outputs=1, name='/generator_1', reuse=False):

        """ We use a U-Net generator here. See unet.py in /models
        """
        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            left_outputs = []

            for level in xrange(self.generator_depth):

                filter_num = int(self.generator_max_filter / (2 ** (self.generator_depth - level)) / self.downsize_filters_factor)

                if level == 0:
                    left_outputs += [Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(inputs)]
                    left_outputs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(left_outputs[level])
                else:
                    left_outputs += [MaxPooling3D(pool_size=self.pool_size)(left_outputs[level - 1])]
                    left_outputs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(left_outputs[level])
                    left_outputs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(left_outputs[level])

                if self.dropout is not None and self.dropout != 0:
                    left_outputs[level] = Dropout(self.dropout)(left_outputs[level])

                if self.batch_norm:
                    left_outputs[level] = BatchNormalization()(left_outputs[level])

            right_outputs = [left_outputs[self.generator_depth - 1]]

            for level in xrange(self.generator_depth):

                filter_num = int(self.generator_max_filter / (2 ** (level)) / self.downsize_filters_factor)

                if level > 0:
                    right_outputs += [UpConvolution(pool_size=self.pool_size)(right_outputs[level - 1])]
                    right_outputs[level] = concatenate([right_outputs[level], left_outputs[self.generator_depth - level - 1]], axis=4)
                    right_outputs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(right_outputs[level])
                    right_outputs[level] = Conv3D(int(filter_num / 2), self.filter_shape, activation=self.activation, padding=self.padding)(right_outputs[level])
                else:
                    continue

                if self.dropout is not None and self.dropout != 0:
                    right_outputs[level] = Dropout(self.dropout)(right_outputs[level])

                if self.batch_norm:
                    right_outputs[level] = BatchNormalization()(right_outputs[level])

            output_layer = Conv3D(int(self.num_outputs), (1, 1, 1))(right_outputs[-1])

            return output_layer

    def build_model(self):
        
        """ A basic implementation of the U-Net proposed in https://arxiv.org/abs/1505.04597
        
            TODO: specify optimizer

            Returns
            -------
            Keras model or tensor
                If input_tensor is provided, this will return a tensor. Otherwise,
                this will return a Keras model.
        """

        # # TODO: Brainstorm better way to specify outputs
        # if self.input_tensor is not None:
        #     return output_layer

        # if self.output_type == 'regression':
        #     self.model = Model(inputs=self.inputs, outputs=output_layer)
        #     self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

        # if self.output_type == 'binary_label':
        #     act = Activation('sigmoid')(output_layer)
        #     self.model = Model(inputs=self.inputs, outputs=act)
        #     self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

        # if self.output_type == 'categorical_label':
        #     act = Activation('softmax')(output_layer)
        #     self.model = Model(inputs=self.inputs, outputs=act)
        #     self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='categorical_crossentropy',
        #                   metrics=['categorical_accuracy'])

        return

    # def discriminator(self, inputs, name='discriminator', scope=None, reuse=False):

    #     convs = []
    #     with tf.variable_scope('discriminator_1/', reuse=tf.AUTO_REUSE) as scope:

    #         if reuse:
    #             scope.reuse_variables()

    #         print(dir(scope))

    #         for level in xrange(self.discriminator_depth):

    #             filter_num = int(self.discriminator_max_filter / (2 ** (self.discriminator_depth - level)) / self.downsize_filters_factor)

    #             if level == 0:
    #                 convs += [Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding, name='dis_conv_1_level' + str(level))(inputs)]
    #                 convs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding=self.padding, name='dis_conv_2_level' + str(level))(convs[level])
    #             else:
    #                 convs += [MaxPooling3D(pool_size=self.pool_size)(convs[level - 1])]
    #                 convs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding, name='dis_conv_1_level' + str(level))(convs[level])
    #                 convs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding=self.padding, name='dis_conv_2_level' + str(level))(convs[level])

    #             if self.dropout is not None and self.dropout != 0:
    #                 convs[level] = Dropout(self.dropout, name='dis_dropout_level' + str(level))(convs[level])

    #             # for t in tf.trainable_variables():
    #                 # print t
    #             print(dir(convs[level]))

    #             if self.batch_norm:
    #                 convs[level] = BatchNormalization(name='dis_batchnorm_level' + str(level))(convs[level])

    #         flatten_layer = Flatten(name='dis_flatten_level' + str(level))(convs[-1])
    #         dense_layer = Dense(64, name='dis_dense_level' + str(level))(flatten_layer)
    #         determination = tf.nn.sigmoid(dense_layer, name='dis_sigmoid_level' + str(level))

    #         return determination

    def predict(self, input_data):

        self.init_sess()

        return self.sess.run(self.generator_1_2, feed_dict={self.generator_input_images_1: input_data})
