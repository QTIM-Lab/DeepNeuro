""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

from deepneuro.models.model import TensorFlowModel
from deepneuro.utilities.util import add_parameter
from deepneuro.models.blocks import generator


class GAN(TensorFlowModel):
    
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

        super(GAN, self).load(kwargs)

        # Generator Parameters
        add_parameter(self, kwargs, 'latent_size', 128)
        add_parameter(self, kwargs, 'depth', 4)

        # Model Parameters
        add_parameter(self, kwargs, 'filter_cap', 128)
        add_parameter(self, kwargs, 'filter_floor', 16)

        add_parameter(self, kwargs, 'generator_max_filter', 128)

        # Discriminator Parameters
        add_parameter(self, kwargs, 'discriminator_depth', 4)
        add_parameter(self, kwargs, 'discriminator_max_filter', 128)

        # Training Parameters
        add_parameter(self, kwargs, 'train_with_GAN', True)
        add_parameter(self, kwargs, 'train_separately', False)

        add_parameter(self, kwargs, 'consistency_weight', 10)  # AKA lambda
        add_parameter(self, kwargs, 'gradient_penalty_weight', 10)

        self.sess = None
        self.init = None

    def get_filter_num(self, depth):

        # This will need to be a bit more complicated; see PGGAN paper.
        if self.max_filter / (2 ** (depth)) <= self.filter_floor:
            return self.filter_floor
        else:
            return min(self.max_filter / (2 ** (depth)), self.filter_cap)

    def train(self, training_data_collection, **kwargs):

        # Outputs
        add_parameter(self, kwargs, 'output_model_filepath')

        # Training Parameters
        add_parameter(self, kwargs, 'num_epochs', 100)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 10)
        add_parameter(self, kwargs, 'training_batch_size', 16)

        self.build_tensorflow_model(self.training_batch_size)
        self.create_data_generators(training_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)
        self.init_sess()

        step = 0
                
        for epoch in range(self.num_epochs):

            step_counter = tqdm(list(range(self.training_steps_per_epoch)), total=self.training_steps_per_epoch, unit="step", desc="Generator Loss:", miniters=1)

            for step in step_counter:

                # Replace with GPU function?
                sample_latent = np.random.normal(size=[self.training_batch_size, self.latent_size])
                reference_data = next(self.training_data_generator)[self.input_data]

                # Optimize!

                _, g_loss = self.sess.run([self.basic_optimizer, self.basic_loss], feed_dict={self.reference_images: reference_data, self.latent: sample_latent})

                self.log([g_loss], headers=['Basic Loss'], verbose=self.hyperverbose)
                step_counter.set_description("Generator Loss: {0:.5f}".format(g_loss))

            self.save_model(self.output_model_filepath)

        return

    def build_tensorflow_model(self, batch_size):

        """ Break it out into functions?
        """

        # Set input/output shapes for reference during inference.
        self.model_input_shape = tuple([batch_size] + list(self.input_shape))
        self.model_output_shape = tuple([batch_size] + list(self.input_shape))

        self.latent = tf.placeholder(tf.float32, [self.training_batch_size, self.latent_size])
        self.reference_images = tf.placeholder(tf.float32, list(self.model_input_shape))

        self.synthetic_images = generator(self, self.latent, depth=self.depth, name='generator')

        self.basic_loss = tf.reduce_mean(tf.square(self.reference_images - self.synthetic_images))

        # Hmmm.. better way to do this? Or at least move to function.
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        # Create save/load operation
        self.saver = tf.train.Saver(self.g_vars + self.d_vars)

        self.basic_optimizer = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.basic_loss, var_list=self.g_vars)

    def load_model(self, input_model_path, batch_size=1):

        self.build_tensorflow_model(batch_size)
        self.init_sess()
        self.saver.restore(self.sess, os.path.join(input_model_path, 'model.ckpt'))

    def predict(self, input_latent=None):

        self.init_sess()

        if input_latent is None:
            input_latent = np.random.normal(size=[self.training_batch_size, self.latent_size])

        return self.sess.run(self.fake_images, feed_dict={self.latent: input_latent})