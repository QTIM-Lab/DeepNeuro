""" This is a vanilla implementation of a generative adversarial network. It includes
    the Wasserstein Gradient-Penalty by default.
"""

import numpy as np
import tensorflow as tf
import os

from deepneuro.models.tensorflow_model import TensorFlowModel
from deepneuro.utilities.util import add_parameter
from deepneuro.models.blocks import generator, discriminator
from deepneuro.models.cost_functions import wasserstein_loss


class GAN(TensorFlowModel):
    
    def load(self, kwargs):

        """ Parameters
            ----------
            latent_size : int, optional
                Size of the latent vector for image synthesis. Default is 128.
            depth : int, optional
                Specifies the number of layers
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        super(GAN, self).load(kwargs)

        # Generator Parameters
        add_parameter(self, kwargs, 'latent_size', 128)
        add_parameter(self, kwargs, 'depth', 4)
        add_parameter(self, kwargs, 'generator_updates', 1)

        # Model Parameters
        add_parameter(self, kwargs, 'filter_cap', 128)
        add_parameter(self, kwargs, 'filter_floor', 16)

        add_parameter(self, kwargs, 'generator_max_filter', 128)

        # Discriminator Parameters
        add_parameter(self, kwargs, 'discriminator_depth', 4)
        add_parameter(self, kwargs, 'discriminator_max_filter', 128)
        add_parameter(self, kwargs, 'discriminator_updates', 1)

        # Loss Parameters
        add_parameter(self, kwargs, 'gradient_penalty_weight', 10)  # For WP

        self.sess = None
        self.init = None

    def get_filter_num(self, depth):

        # This will need to be a bit more complicated; see PGGAN paper.
        if self.max_filter / (2 ** (depth)) <= self.filter_floor:
            return self.filter_floor
        else:
            return min(self.max_filter / (2 ** (depth)), self.filter_cap)

    def process_step(self, step_counter):

        # Replace with GPU function?
        sample_latent = np.random.normal(size=[self.training_batch_size, self.latent_size])
        reference_data = next(self.training_data_generator)[self.input_data]

        # Optimize!

        _, g_loss = self.sess.run([self.opti_G, self.G_loss], feed_dict={self.reference_images: reference_data, self.latent: sample_latent})
        _, d_loss, d_origin_loss = self.sess.run([self.opti_D, self.D_loss, self.d_origin_loss], feed_dict={self.reference_images: reference_data, self.latent: sample_latent})

        # This is a little weird -- it only records loss on discriminator steps.
        self.log([g_loss, d_loss, d_origin_loss], headers=['Generator Loss', 'WP Discriminator Loss', 'Discriminator Loss'], verbose=self.hyperverbose)
        step_counter.set_description("Generator Loss: {0:.5f}".format(g_loss) + " Discriminator Loss: {0:.5f}".format(d_loss))

        return

    def build_tensorflow_model(self, batch_size):

        """ Break it out into functions?
        """

        # Set input/output shapes for reference during inference.
        self.model_input_shape = tuple([batch_size] + list(self.input_shape))
        self.model_output_shape = tuple([batch_size] + list(self.input_shape))

        self.latent = tf.placeholder(tf.float32, [None, self.latent_size])
        self.reference_images = tf.placeholder(tf.float32, [None] + list(self.model_input_shape)[1:])
        self.synthetic_images = generator(self, self.latent, depth=self.depth, name='generator')

        self.discriminator_real, self.discriminator_real_logits = discriminator(self, self.reference_images, depth=self.depth + 1, name='discriminator')
        self.discriminator_fake, self.discriminator_fake_logits = discriminator(self, self.synthetic_images, depth=self.depth + 1, name='discriminator', reuse=True)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.saver = tf.train.Saver(self.g_vars + self.d_vars)

        self.calculate_losses()

        if self.hyperverbose:
            self.model_summary()

    def calculate_losses(self):

        self.D_loss, self.G_loss, self.D_origin_loss = wasserstein_loss(self, discriminator, self.discriminator_fake_logits, self.discriminator_real_logits, self.synthetic_images, self.real_images, gradient_penalty_weight=self.gradient_penalty_weight, name='discriminator', dim=self.dim)

        # A little sketchy. Attempting to make variable loss functions extensible later.
        self.D_loss = self.D_loss[0]
        self.G_loss = self.G_loss[0]
        self.D_origin_loss = self.D_origin_loss[0]

        # Create Optimizers
        self.opti_D = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        self.opti_G = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.G_loss, var_list=self.g_vars)

    def load_model(self, input_model_path, batch_size=1):

        self.build_tensorflow_model(batch_size)
        self.init_sess()
        self.saver.restore(self.sess, os.path.join(input_model_path, 'model.ckpt'))

    def predict(self, sample_latent=None, batch_size=1):

        self.init_sess()

        if sample_latent is None:
            sample_latent = np.random.normal(size=[batch_size, self.latent_size])

        return self.sess.run(self.synthetic_images, feed_dict={self.latent: sample_latent})

    def log_variables(self):

        super(GAN, self).log_variables()

        return