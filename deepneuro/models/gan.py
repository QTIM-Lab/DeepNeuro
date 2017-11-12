""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

from keras.engine import Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Dropout, BatchNormalization
from keras.optimizers import Nadam
from keras.layers.merge import concatenate

from deepneuro.models.cost_functions import dice_coef_loss, dice_coef
from deepneuro.models.model import DeepNeuroModel, UpConvolution
from deepneuro.utilities.conversion import round_up
from deepneuro.models.dn_ops import batch_norm, relu, tanh, leaky_relu, dense, reshape, sigmoid

import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

class GAN(DeepNeuroModel):
    
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

        if 'depth' in kwargs:
            self.depth = kwargs.get('depth')
        else:
            self.depth = 4

        if 'max_filter' in kwargs:
            self.max_filter = kwargs.get('max_filter')
        else:
            self.max_filter = 512

        if 'vector_size' in kwargs:
            self.vector_size = kwargs.get('vector_size')
        else:
            self.vector_size = 64

    def generator(self, vector):

        with tf.variable_scope("generator") as scope:

            convs = []
            output_shapes = []
            filter_nums = []

            for level in xrange(self.depth):

                # 
                if level == 0:
                    output_shapes += [self.output_dims]
                    filter_nums += [self.max_filter]
                else:
                    output_shapes = [[round_up(dim, 2 ** level) for dim in self.output_dims]] + output_shapes
                    filter_nums += [round_up(self.max_filter, 2 ** level)]

            dense_layer = dense(vector, filter_nums[0] * np.prod(output_shapes[0]), with_w=True)
            convs[0] = relu()(batch_norm()((reshape()(dense_layer[0], [-1, output_shapes[0] + filter_nums[0]]))))

            for level in xrange(1, self.depth):

                convs += [deconv2d(convs[level-1][0], [self.batch_size] + output_shapes[level] + filter_nums[level], with_w=True)]
                
                if self.batch_norm:
                    convs[level][0] = batch_norm()(convs[level][0])

                if level == self.depth - 1:
                    convs[level][0] = tanh()(convs[level][0])
                else:
                    convs[level][0] = relu()(convs[level][0])

            return convs[-1][0]

    def discriminator(self, image, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            convs = []
            output_shapes = []
            filter_nums = []

            for level in xrange(self.depth):

                if level == 0:
                    filter_nums += [self.max_filter]
                else:
                    filter_nums = [round_up(self.max_filter, 2 ** level)] + filter_nums

            for level in xrange(1, self.depth):

                convs += [conv2d(convs[level], self.filter_nums[level])]
                
                if self.batch_norm:
                    convs[level] = batch_norm()(convs[level])

                convs[level] = leaky_relu()(convs[level])

            dense = reshape()(convs[-1], [self.batch_size, -1])
            dense = dense()(dense, 1)
            dense = sigmoid()(dense)

            return dense

    # def sampler(self, z, y=None):
    #     with tf.variable_scope("generator") as scope:
    #       scope.reuse_variables()

    #       if not self.y_dim:
    #         s_h, s_w = self.output_height, self.output_width
    #         s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
    #         s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
    #         s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
    #         s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

    #         # project `z` and reshape
    #         h0 = tf.reshape(
    #             linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
    #             [-1, s_h16, s_w16, self.gf_dim * 8])
    #         h0 = tf.nn.relu(self.g_bn0(h0, train=False))

    #         h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
    #         h1 = tf.nn.relu(self.g_bn1(h1, train=False))

    #         h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
    #         h2 = tf.nn.relu(self.g_bn2(h2, train=False))

    #         h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
    #         h3 = tf.nn.relu(self.g_bn3(h3, train=False))

    #         h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

    #         return tf.nn.tanh(h4)
    #       else:
    #         s_h, s_w = self.output_height, self.output_width
    #         s_h2, s_h4 = int(s_h/2), int(s_h/4)
    #         s_w2, s_w4 = int(s_w/2), int(s_w/4)

    #         # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
    #         yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
    #         z = concat([z, y], 1)

    #         h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
    #         h0 = concat([h0, y], 1)

    #         h1 = tf.nn.relu(self.g_bn1(
    #             linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
    #         h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
    #         h1 = conv_cond_concat(h1, yb)

    #         h2 = tf.nn.relu(self.g_bn2(
    #             deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
    #         h2 = conv_cond_concat(h2, yb)

    #         return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


    def build_model(self):
        
        """ A basic implementation of the U-Net proposed in https://arxiv.org/abs/1505.04597
        
            TODO: specify optimizer

            Returns
            -------
            Keras model or tensor
                If input_tensor is provided, this will return a tensor. Otherwise,
                this will return a Keras model.
        """

        self.channels = 1

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.channels]
        else:
            image_dims = [self.input_height, self.input_width, self.channels]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + list(self.inputs.get_shape()), name='real_images')

        self.vectors = tf.placeholder(tf.float32, [None, self.vector_size], name='vectors')

        self.G                  = self.generator(self.vectors)
        self.D, self.D_logits   = self.discriminator(self.inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        
        # self.sampler            = self.sampler(self.z, self.y)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        t_vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.saver = tf.train.Saver()

        return self.model

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model'], **kwargs):

        self.batch_size = training_batch_size

        if training_steps_per_epoch is None:
            training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1

        training_data_generator = training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=training_batch_size)

        sample_vector = np.random.uniform(-1, 1, size=(self.batch_size, self.vector_size))

        discriminator_optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        generator_optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.g_loss, var_list=self.g_vars)

        start_time = time.time()

        for epoch in xrange(num_epochs):

            for batch_idx in xrange(training_steps_per_epoch):

                batch_vector = np.random.uniform(-1, 1, [self.batch_size, self.vector_size]).astype(np.float32)
                batch_images = next(training_data_generator)

                # Update D network
                _, summary_str = self.sess.run([discriminator_optimizer, self.d_sum], feed_dict={ self.inputs: batch_images, self.z: batch_vector })

                # Update G network (twice to help training)
                _, summary_str = self.sess.run([generator_optimizer, self.g_sum], feed_dict={ self.vectors: batch_vector })
                _, summary_str = self.sess.run([generator_optimizer, self.g_sum], feed_dict={ self.vectors: batch_vector })
          
                errD_fake = self.d_loss_fake.eval({ self.vectors: batch_z })
                errD_real = self.d_loss_real.eval({ self.vectors: batch_images })
                errG = self.g_loss.eval({self.vectors: batch_vector})

                if batch_idx % 50 == 0:
                    print batch_idx
                #     samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],feed_dict={self.z: sample_z, self.inputs: sample_inputs})
                #     save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                #     print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
              
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, batch_idx, batch_idx, time.time() - start_time, errD_fake+errD_real, errG))


    def save(self, epoch):
        model_name = 'DCGAN_' + str(epoch) + '.model'

        self.saver.save(self.sess, model_name)

        