""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

import tensorflow as tf
import scipy
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lycon
import csv

from skimage import draw

from deepneuro.models.model import TensorFlowModel
from deepneuro.models.cost_functions import wasserstein_loss
from deepneuro.models.dn_ops import DnConv, DnAveragePooling, pixel_norm, dense, minibatch_state_concat, leaky_relu, upscale, downscale
from deepneuro.utilities.util import add_parameter


class ProgressiveGAN(TensorFlowModel):
    
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

        super(ProgressiveGAN, self).load(kwargs)

        add_parameter(self, kwargs, 'dim', 2)

        # Generator Parameters
        add_parameter(self, kwargs, 'generator_depth', 4)
        add_parameter(self, kwargs, 'generator_max_filter', 4096)
        add_parameter(self, kwargs, 'generator_updates', 1)

        # Discriminator Parameters
        add_parameter(self, kwargs, 'discriminator_depth', 4)
        add_parameter(self, kwargs, 'discriminator_max_filter', 4096)
        add_parameter(self, kwargs, 'discriminator_updates', 1)

        # GAN Parameters
        add_parameter(self, kwargs, 'latent_size', 128)

        # PGGAN Parameters
        add_parameter(self, kwargs, 'starting_depth', 1)
        add_parameter(self, kwargs, 'depth', 9)
        add_parameter(self, kwargs, 'transition', True)
        add_parameter(self, kwargs, 'filter_cap', 128)
        add_parameter(self, kwargs, 'filter_floor', 16)
        add_parameter(self, kwargs, 'logging_directory', None)

        # Wasserstein Parameters
        add_parameter(self, kwargs, 'gradient_penalty_weight', 10)

        # Conditional Parameters
        add_parameter(self, kwargs, 'classify', None)

        # Training Parameters
        add_parameter(self, kwargs, 'training_batch_size', 16)

        # Derived Parameters
        self.channels = self.input_shape[-1]

        self.sess = None
        self.init = None

    def get_filter_num(self, depth, max_filter):

        # This will need to be a bit more complicated; see PGGAN paper.
        if max_filter / (2 ** (depth)) <= self.filter_floor:
            return self.filter_floor
        else:
            return min(max_filter / (2 ** (depth)), self.filter_cap)

    # @profile
    def train(self, training_data_collection, validation_data_collection=None, **kwargs):

        # Outputs
        add_parameter(self, kwargs, 'output_model_filepath', 'pgan_model')

        # Training Parameters
        add_parameter(self, kwargs, 'num_epochs', 100)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 10)
        add_parameter(self, kwargs, 'training_batch_size', 16)
        add_parameter(self, kwargs, 'learning_rate', 0.0001)

        # self.create_data_generators(training_data_collection, validation_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)

        # Create necessary directories
        for work_dir in [self.output_model_filepath, self.logging_directory]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        with open('/mnt/nas2/data/Personal/Andrew/Interp_GAN/backup/path_filenames.csv', 'rb') as f:
            reader = csv.reader(f)
            input_files = list(reader)[0]

        np.random.shuffle(input_files)
        total_images = len(input_files)
        image_num = 0

        print range(int(np.ceil((self.starting_depth - 1) / 2.)), (self.depth * 2) - 1)
        for training_stage in range(int(np.ceil((self.starting_depth - 1) / 2.)), (self.depth * 2) - 1):

            print('TRAINING_STAGE', training_stage)

            if (training_stage % 2 == 0):
                self.transition = False
            else:
                self.transition = True

            current_depth = np.ceil((training_stage + 2) / 2.)
            previous_depth = np.ceil((training_stage + 1) / 2.)
            self.progressive_depth = int(current_depth)
            print 'DEPTHS', current_depth, previous_depth

            self.samples_dir = os.path.join(self.logging_directory, 'sample_' + str(current_depth) + '_' + str(self.transition))
            if not os.path.exists(self.samples_dir):
                os.mkdir(self.samples_dir)

            current_model_path = os.path.join(self.output_model_filepath, str(current_depth), 'model.ckpt')
            if not os.path.exists(os.path.dirname(current_model_path)):
                os.mkdir(os.path.dirname(current_model_path))
            previous_model_path = os.path.join(self.output_model_filepath, str(previous_depth), 'model.ckpt')

            self.build_tensorflow_model(self.training_batch_size)
            self.init_sess(new_sess=True)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.logging_directory, self.sess.graph)

            if self.transition:
                self.r_saver.restore(self.sess, previous_model_path)
                self.rgb_saver.restore(self.sess, previous_model_path)
            elif self.progressive_depth > 1:
                self.saver.restore(self.sess, previous_model_path)

            # realbatch_array = next(self.training_data_generator)[0]
            # realbatch_array = self.sess.run(self.input_volumes, feed_dict={self.raw_volumes: realbatch_array})
            # sample_latent = np.random.normal(size=[self.training_batch_size, self.latent_size])

            for epoch in range(self.num_epochs):

                # self.checkpoint(realbatch_array, sample_latent, epoch, 0)

                for step in range(self.training_steps_per_epoch):
                    
                    print epoch, step

                    # Update Discriminator
                    for i in range(self.discriminator_updates):

                        sample_latent = np.random.normal(size=[self.training_batch_size, self.latent_size])

                        # print 'about to load data'
                        # realbatch_array = next(self.training_data_generator)[0]

                        images = []
                        for i in range(self.training_batch_size):
                            input_file = input_files[image_num % total_images]
                            path_data = lycon.load(input_file)
                            mask_filepath = os.path.join('/mnt/nas3/bigdatasets/Desuto/binary_masks_predictions/train', os.path.basename(os.path.dirname(os.path.dirname(input_file))), os.path.basename(os.path.dirname(input_file)), os.path.basename(input_file)[0:-4] + '_pred-mask.png')
                            mask_data = lycon.load(mask_filepath)

                            path_data = np.asarray(path_data)[..., ::-1].astype(float) / 127.5 - 1
                            mask_data = (np.asarray(mask_data)[..., 0] > 0).astype(float)[..., np.newaxis]
                            mask_data[mask_data == 0] = -1              

                            images += [np.concatenate([path_data, mask_data], axis=2)]
                            image_num += 1

                        realbatch_array = np.stack(images, axis=0)

                        # print 'about to resize data'

                        # arr = np.zeros((20, 256, 256, 4))
                        # rr, cc = draw.circle(128, 128, radius=90, shape=(256, 256))
                        # arr[:, rr, cc, :] = 1
                        # arr += np.random.uniform(-1, 0, (20, 256, 256, 4))
                        # realbatch_array = arr

                        realbatch_array = self.sess.run(self.input_volumes, feed_dict={self.raw_volumes: realbatch_array})
                        # print 'about to optimize'

                        if self.transition:
                            realbatch_array = self.sess.run(self.real_images, feed_dict={self.images: realbatch_array})

                        self.sess.run(self.opti_D, feed_dict={self.images: realbatch_array, self.latent: sample_latent})

                    # Update Generator
                    for i in range(self.discriminator_updates):
                        self.sess.run(self.opti_G, feed_dict={self.latent: sample_latent})

                    # the alpha of fake_in process
                    if self.transition:
                        self.sess.run(self.alpha_transition_assign, feed_dict={self.step_pl: (epoch * float(self.training_steps_per_epoch)) + (step + 1)})

                    D_loss_WP, D_loss, G_loss, transition = self.sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_transition], feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    self.log([D_loss_WP, D_loss, G_loss, transition], headers=['Dis-WP Loss', 'Dis Loss', 'Gen Loss', 'Alpha'], verbose=self.hyperverbose)

                self.saver.save(self.sess, current_model_path)

                self.checkpoint(realbatch_array, sample_latent, epoch, 0)

            save_path = self.saver.save(self.sess, current_model_path)
            print "Model saved in file: %s" % save_path

            tf.reset_default_graph()

    def build_tensorflow_model(self, batch_size):

        # Set input/output shapes for reference during inference.
        self.model_input_shape = tuple([batch_size] + list(self.input_shape))
        self.model_output_shape = tuple([batch_size] + list(self.input_shape))

        self.latent = tf.placeholder(tf.float32, [self.training_batch_size, self.latent_size])

        # Derived Parameters
        self.output_size = pow(2, self.progressive_depth + 1)
        self.zoom_level = self.progressive_depth
        self.images = tf.placeholder(tf.float32, [self.training_batch_size, self.output_size, self.output_size, self.channels]
        self.alpha_transition = tf.Variable(initial_value=0.0, trainable=False, name='alpha_transition')

        self.fake_images = self.generator(self.latent, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, name='generator')

        self.Q_generated_real, _, _, self.D_pro_logits = self.discriminator(self.images, reuse=False, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, name='discriminator')
        self.Q_generated_fake, _, _, self.G_pro_logits = self.discriminator(self.fake_images, reuse=True, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, name='discriminator')

        # Loss functions
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # Gradient Penalty from Wasserstein GAN GP, I believe? Check on it --andrew
        # Also investigate more what's happening here --andrew
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.training_batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, _, _, discri_logits = self.discriminator(interpolates, reuse=True, progressive_depth=self.progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition, name='discriminator')
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # Some sort of norm from papers, check up on it. --andrew
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        # Update Loss functions..
        self.D_origin_loss = self.D_loss
        self.D_loss += 10 * self.gradient_penalty
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

        # Data Loading Tools
        self.low_images = upscale(downscale(self.images, 2), 2)
        self.real_images = self.alpha_transition * self.images + (1 - self.alpha_transition) * self.low_images

        self.log_vars = []
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        # Hmmm.. better way to do this? Or at least move to function.
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        # save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'discriminator_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'generator_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'discriminator_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'generator_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)
        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        # for layer in self.d_vars + self.g_vars:
            # print layer

        # Create fade-in (transition) parameters.
        self.step_pl = tf.placeholder(tf.float32, shape=None)
        self.alpha_transition_assign = self.alpha_transition.assign(self.step_pl / (self.num_epochs * self.training_steps_per_epoch))

        # Create Optimizers
        self.opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        self.opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        max_downscale = np.floor(math.log(self.model_input_shape[1], 2))
        downscale_factor = 2 ** max_downscale / (2 ** (self.progressive_depth + 1))
        print("DOWNSCALE FACTOR", downscale_factor)
        self.raw_volumes = tf.placeholder(tf.float32, self.model_input_shape)
        self.input_volumes = downscale(self.raw_volumes, downscale_factor)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def generator(self, latent_var, progressive_depth=1, name=None, transition=False, alpha_transition=0.0, reuse=False):

        print 'GENERATOR', progressive_depth

        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            convs = []

            convs += [tf.reshape(latent_var, [self.training_batch_size, 1, 1, self.latent_size])]

            convs[-1] = pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(1, self.generator_max_filter), kernel_size=(4, 4), stride_size=(1, 1), padding='Other', name='generator_n_1_conv', dim=self.dim)))

            convs += [tf.reshape(convs[-1], [self.training_batch_size, 4, 4, self.get_filter_num(1, self.generator_max_filter)])]
            convs[-1] = pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(1, self.generator_max_filter), stride_size=(1, 1), name='generator_n_2_conv', dim=self.dim)))

            for i in range(progressive_depth - 1):

                if i == progressive_depth - 2 and transition:  # redundant conditions? --andrew
                    #To RGB
                    transition_conv = DnConv(convs[-1], output_dim=self.channels, kernel_size=(1, 1), stride_size=(1, 1), name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=self.dim)
                    transition_conv = upscale(transition_conv, 2)

                convs += [upscale(convs[-1], 2)]
                convs[-1] = pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(i + 1, self.generator_max_filter), stride_size=(1, 1), name='generator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=self.dim)))

                convs += [pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(i + 1, self.generator_max_filter), stride_size=(1, 1), name='generator_n_conv_2_{}'.format(convs[-1].shape[1]), dim=self.dim)))]

            #To RGB
            convs += [DnConv(convs[-1], output_dim=self.channels, kernel_size=(1, 1), stride_size=(1, 1), name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=self.dim)]

            if transition:
                convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

            # for conv in convs:
                # print conv

            return convs[-1]

    def load_model(self, input_model_path, batch_size=1):

        self.build_tensorflow_model(batch_size)
        self.init_sess()
        self.saver.restore(self.sess, os.path.join(input_model_path, 'model.ckpt'))

    def checkpoint(self, input_data, input_latent, epoch, step):

        input_data = np.clip(input_data, -1, 1)
        self.save_images(input_data[0:self.training_batch_size], [2, self.training_batch_size / 2], '{}/{:02d}_real.png'.format(self.samples_dir, (1 + epoch) * (1 + step)))

        fake_image = self.sess.run(self.fake_images, feed_dict={self.images: input_data, self.latent: input_latent})
        fake_image = np.clip(fake_image, -1, 1)
        self.save_images(fake_image[0:self.training_batch_size], [2, self.training_batch_size / 2], '{}/{:02d}_train.png'.format(self.samples_dir, (1 + epoch) * (1 + step)))

    def merge(self, images, size, channels=3):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], channels))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

        return img

    def imsave(self, images, size, path):

        if images.shape[-1] > 4:
            for channel in xrange(4):
                new_path = path[0:-4] + '_' + str(channel) + '.png'
                lycon.save(new_path, self.merge(images[..., channel][..., np.newaxis], size))
            return lycon.save(path, self.merge(images[..., 4:], size))

        elif images.shape[-1] == 3:
            return lycon.save(path, self.merge(images, size))

        elif images.shape[-1] == 1:
            lycon.save(path, np.squeeze(self.merge(images[:, :, :, 0][:, :, :, np.newaxis], size, channels=1)))

        else:
            lycon.save(path, self.merge(images[:, :, :, :3], size))
            new_path = path[0:-4] + '_mask.png'
            return lycon.save(new_path, np.squeeze(self.merge(images[..., 3][..., np.newaxis], size, channels=1)))

    def inverse_transform(self, image):
        return ((image + 1.) * 127.5).astype(np.uint8)

    def save_images(self, images, size, image_path):
        data = self.inverse_transform(images)
        print('Saving data, shape:', data.shape)
        return self.imsave(data, size, image_path)

    def predict(self, input_data):

        self.init_sess()

        return self.sess.run(self.generator_1_2_real, feed_dict={self.generator_input_images_1: input_data})
