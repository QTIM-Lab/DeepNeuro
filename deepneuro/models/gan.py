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
from deepneuro.models.dn_ops import batch_norm, relu, tanh, leaky_relu, dense, reshape, sigmoid, conv2d, deconv2d, conv3d, deconv3d

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import csv

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

        self.batch_size = 32
        self.input_shape = (64,64,8,1)

    def generator(self, vector):

        with tf.variable_scope("generator") as scope:

            convs = []
            output_shapes = []
            filter_nums = []
            kernel_shapes = [(2,2,1), (4,4,2), (8,8,2), (8,8,2)]

            for level in xrange(self.depth):

                if level == 0:
                    output_shapes += [list(self.input_shape[:-1])]
                    filter_nums += [self.max_filter]
                else:
                    output_shapes = [[round_up(dim, 2 ** level) for dim in self.input_shape[:-1]]] + output_shapes
                    filter_nums += [round_up(self.max_filter, 2 ** level)]

            filter_nums[-1] = self.input_shape[-1]

            dense_layer = dense(vector, filter_nums[0] * np.prod(output_shapes[0]), with_w=True)
            convs += [leaky_relu((batch_norm()((reshape()(dense_layer[0], [-1] + output_shapes[0] + [filter_nums[0]])))))]

            for level in xrange(1, self.depth):

                if level == self.depth - 1:
                    convs += [deconv3d(convs[-1], [self.batch_size] + output_shapes[level] + [filter_nums[level-1]], with_w=False, name='g_deconv3d_' + str(level))]
                    convs[-1] = leaky_relu((convs[-1]))
                    convs[-1] = tf.nn.dropout(convs[-1], .5)
                    convs[-1] = batch_norm()(convs[-1])

                    convs += [conv3d(convs[-1], filter_nums[level], name='g_conv3d_' + str(level), stride_size=(1,1,1))]
                    # convs[-1] = tanh()(convs[-1])
                else:
                    convs += [deconv3d(convs[-1], [self.batch_size] + output_shapes[level] + [filter_nums[level]], with_w=False, name='g_deconv3d_' + str(level))]
                    convs[-1] = leaky_relu((convs[-1]))
                    convs[-1] = tf.nn.dropout(convs[-1], .5)
                    convs[-1] = batch_norm()(convs[-1])

                    # convs += [conv3d(convs[-1], filter_nums[level]/2, name='g_conv3d_' + str(level), padding='SAME', stride_size=(1,1,1))]
                    # convs[-1] = leaky_relu((convs[-1]))
                    # convs[-1] = tf.nn.dropout(convs[-1], .5)
                    # convs[-1] = batch_norm()(convs[-1])

            for i in convs:
                print 'GENERATOR', i

            return convs[-1]

    def discriminator(self, image, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            convs = []
            output_shapes = []
            filter_nums = []
            kernel_shapes = [(8,8,2), (4,4,2), (2,2,2), (2,2,1)]

            for level in xrange(self.depth):

                if level == 0:
                    filter_nums += [self.max_filter]
                else:
                    filter_nums = [round_up(self.max_filter, 2 ** level)] + filter_nums

            print image

            for level in xrange(self.depth):

                print 'DISCRIMINATOR', convs

                if level == 0:
                    convs += [conv3d(image, filter_nums[level], name='d_conv3d_' + str(level), kernel_size=kernel_shapes[level], padding='VALID', stride_size=(2,2,2))]
                else:
                    convs += [conv3d(convs[-1], filter_nums[level], name='d_conv3d_' + str(level), kernel_size=kernel_shapes[level], padding='VALID', stride_size=(2,2,2))]
                
                convs[-1] = leaky_relu((convs[-1]))

                if self.batch_norm:
                    convs[-1] = batch_norm()(convs[-1])

            dense_layer = reshape()(convs[-1], [self.batch_size, -1])
            dense_layer = dense(dense_layer, 1)

            return sigmoid()(dense_layer), dense_layer

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

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + list(self.input_shape), name='real_images')
        self.vectors = tf.placeholder(tf.float32, [None, self.vector_size], name='vectors')

        self.true_labels = tf.placeholder(tf.float32, [self.batch_size, 1], name='true_labels')
        self.false_labels = tf.placeholder(tf.float32, [self.batch_size, 1], name='false_labels')

        self.G                  = self.generator(self.vectors)
        self.D, self.D_logits   = self.discriminator(self.inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        
        # self.sampler            = self.sampler(self.z, self.y)

        def gaussian_noise_layer(input_layer, std=.2):
            noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
            return input_layer + noise

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(gaussian_noise_layer(self.D_logits), self.true_labels))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(gaussian_noise_layer(self.D_logits_), self.false_labels))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(gaussian_noise_layer(self.D_logits_), self.true_labels))

        t_vars = tf.trainable_variables()
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.saver = tf.train.Saver()

        return self.model

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model'], **kwargs):

        with tf.Session() as sess:

            self.sess = sess
            self.batch_size = training_batch_size

            if training_steps_per_epoch is None:
                training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1

            training_data_generator = training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=training_batch_size)

            sample_vector = np.random.uniform(-1, 1, size=(self.batch_size, self.vector_size))

            discriminator_optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.d_loss, var_list=self.d_vars)
            generator_optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.g_loss, var_list=self.g_vars)

            try:
              tf.global_variables_initializer().run()
            except:
              tf.initialize_all_variables().run()

            start_time = time.time()

            try:
                    # Save output.
                with open('loss_log.csv', 'ab') as writefile:
                    csvfile = csv.writer(writefile, delimiter=',')
                    csvfile.writerow(['g_loss', 'd_loss_real', 'd_loss_fake', 'd_loss'])
                    for epoch in xrange(num_epochs):

                        for batch_idx in xrange(training_steps_per_epoch):

                            batch_vector = np.random.uniform(-1, 1, size=(self.batch_size, self.vector_size)).astype(np.float32)
                            batch_images = next(training_data_generator)[0]

                            # batch_images = ((batch_images - np.min(batch_images)) / (np.max(batch_images) - np.min(batch_images))) * 2 - 1

                            if batch_idx % 10 == 0:
                                true = np.random.normal(0, 0.3, [self.batch_size, 1]).astype(np.float32)
                                false = np.random.normal(0.7, 1.3, [self.batch_size, 1]).astype(np.float32)
                            else:
                                false = np.random.normal(0, 0.3, [self.batch_size, 1]).astype(np.float32)
                                true = np.random.normal(0.7, 1.3, [self.batch_size, 1]).astype(np.float32)

                            # Update D network
                            _ = self.sess.run([discriminator_optimizer], feed_dict={ self.inputs: batch_images, self.vectors: batch_vector, self.true_labels: true, self.false_labels: false })

                            # Update G network (twice to help training)
                            _ = self.sess.run([generator_optimizer], feed_dict={ self.vectors: batch_vector, self.true_labels: true, self.false_labels: false })
                            # _ = self.sess.run([generator_optimizer], feed_dict={ self.vectors: batch_vector, self.true_labels: true, self.false_labels: false })

                            errD_fake = self.d_loss_fake.eval({ self.vectors: batch_vector, self.true_labels: true, self.false_labels: false })
                            errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.true_labels: true, self.false_labels: false})
                            errG = self.g_loss.eval({self.vectors: batch_vector, self.true_labels: true, self.false_labels: false})

                            # print 'FAKE ERR', errD_fake, 'REAL ERR', errD_real, 'G ERR', errG

                            # if batch_idx % 50 == 0:
                            #     print batch_idx
                            #     samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],feed_dict={self.z: sample_z, self.inputs: sample_inputs})
                            #     save_images(samples, image_manifold_size(samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            #     print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                          
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, batch_idx, batch_idx, time.time() - start_time, errD_fake+errD_real, errG))
                        self.save(epoch)
                        # csvfile.writerow([errG, errD_real, errD_fake, errD_fake+errD_real])

                        batch_vector = np.random.uniform(-1, 1, [self.batch_size, self.vector_size]).astype(np.float32)
                        test_output = self.sess.run([self.G], feed_dict={ self.vectors: batch_vector })
                        for i in xrange(test_output[0].shape[0]):
                            data = test_output[0][i,...,0]
                            save_numpy_2_nifti(data, np.eye(4), 'other_gan_test_' + str(i) + '.nii.gz')
                            if epoch == 0:
                                save_numpy_2_nifti(batch_images[i,...,0], np.eye(4), 'sample_patch_' + str(i) + '.nii.gz')

            except KeyboardInterrupt:
                pass
            #     batch_vector = np.random.uniform(-1, 1, [self.batch_size, self.vector_size]).astype(np.float32)
            #     test_output = self.sess.run([self.G], feed_dict={ self.vectors: batch_vector })
            #     for i in xrange(test_output[0].shape[0]):
            #         data = test_output[0][i,...,0]
            #         save_numpy_2_nifti(data, np.eye(4), 'other_gan_test_' + str(i) + '.nii.gz')
            #         save_numpy_2_nifti(data, np.eye(4), 'sample_patch_' + str(i) + '.nii.gz')


    def save(self, epoch):
        model_name = 'DCGAN_final_small' + str(epoch) + '.model'

        self.saver.save(self.sess, model_name)

        