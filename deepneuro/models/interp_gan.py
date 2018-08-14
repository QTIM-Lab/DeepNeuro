""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

import numpy as np
import tensorflow as tf
import os
import keras

from deepneuro.models.model import TensorFlowModel, load_old_model
from deepneuro.models.unet import UNet
from deepneuro.utilities.util import add_parameter
from deepneuro.models.blocks import generator, discriminator


class InterpGAN(TensorFlowModel):
    
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

        super(InterpGAN, self).load(kwargs)

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

        add_parameter(self, kwargs, 'input_tensor_name', 'input_1')
        add_parameter(self, kwargs, 'activated_tensor_name', 'downsampling_conv_1_1/BiasAdd')
        add_parameter(self, kwargs, 'filter_num', 0)

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

        # _, b_loss, a_loss = self.sess.run([self.combined_optimizer, self.basic_loss, self.activation_loss], feed_dict={self.latent: sample_latent, self.reference_images: reference_data})

        _, g_loss, a_loss, total_g_loss = self.sess.run([self.opti_G, self.G_activation_loss, self.activation_loss, self.G_loss], feed_dict={self.latent: sample_latent, self.reference_images: reference_data})

        _, d_loss = self.sess.run([self.opti_D, self.D_loss], feed_dict={self.latent: sample_latent, self.reference_images: reference_data})

        # self.log([b_loss], headers=['Basic Loss'], verbose=self.hyperverbose)
        # step_counter.set_description("Activation Loss: {0:.5f}".format(a_loss) + " Basic Loss: {0:.5f}".format(b_loss))
        step_counter.set_description("Activation Loss: {0:.5f}".format(a_loss) + " G Loss: {0:.5f}".format(g_loss) + " D Loss: {0:.5f}".format(d_loss) + " Big_G  Loss: {0:.5f}".format(total_g_loss))

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

        _, _, _, self.discriminator_real_logits = discriminator(self, self.reference_images, depth=self.depth + 1, name='discriminator')
        _, _, _, self.discriminator_fake_logits = discriminator(self, self.synthetic_images, depth=self.depth + 1, name='discriminator', reuse=True)

        self.basic_loss = tf.reduce_mean(tf.square(self.reference_images - self.synthetic_images))

        # Loss functions
        self.D_loss = tf.reduce_mean(self.discriminator_fake_logits) - tf.reduce_mean(self.discriminator_real_logits)
        self.G_loss = -tf.reduce_mean(self.discriminator_fake_logits)

        # Gradient Penalty from Wasserstein GAN GP, I believe? Check on it --andrew
        # Also investigate more what's happening here --andrew
        self.differences = self.synthetic_images - self.reference_images
        self.alpha = tf.random_uniform(shape=[tf.shape(self.differences)[0], 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.reference_images + (self.alpha * self.differences)
        _, _, _, discri_logits = discriminator(self, interpolates, reuse=True, depth=self.depth + 1, name='discriminator')
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # Some sort of norm from papers, check up on it. --andrew
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        # Update Loss functions..
        self.D_origin_loss = self.D_loss
        self.D_loss += 10 * self.gradient_penalty
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.discriminator_real_logits - 0.0))

        # vgg_model = tf.keras.applications.VGG19(include_top=False,
        #                                     weights='imagenet',
        #                                     input_tensor=self.synthetic_images,
        #                                     input_shape=(64, 64, 3),
        #                                     pooling=None,
        #                                     classes=1000)
        # print(vgg_model)

        # self.load_reference_model()

        input_tensor = keras.layers.Input(tensor=self.synthetic_images, shape=self.input_shape)

        model_parameters = {'input_shape': self.input_shape,
                    'downsize_filters_factor': 1,
                    'pool_size': (2, 2), 
                    'kernel_size': (3, 3), 
                    'dropout': 0, 
                    'batch_norm': True, 
                    'initial_learning_rate': 0.00001, 
                    'output_type': 'binary_label',
                    'num_outputs': 1, 
                    'activation': 'relu',
                    'padding': 'same', 
                    'implementation': 'keras',
                    'depth': 3,
                    'max_filter': 128,
                    'stride_size': (1, 1),
                    'input_tensor': input_tensor}

        unet_output = UNet(**model_parameters)
        unet_model = keras.models.Model(input_tensor, unet_output.output_layer)
        unet_model.load_weights('retinal_seg_weights.h5')

        if self.hyperverbose:
            self.model_summary()

        # self.find_layers(['sampling'])

        self.activated_tensor = self.grab_tensor(self.activated_tensor_name)
        print self.activated_tensor
        self.activated_tensor = tf.stack([self.activated_tensor[..., self.filter_num]], axis=-1)
        print self.activated_tensor
        # self.input_tensor = self.grab_tensor(self.input_tensor_name)

        self.activation_loss = -1 * tf.reduce_mean(self.activated_tensor)
        self.activaton_graidents = tf.gradients(self.activation_loss, self.synthetic_images)
        print self.activaton_graidents

        # Hmmm.. better way to do this? Or at least move to function.
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        # Create save/load operation
        self.saver = tf.train.Saver(self.g_vars + self.d_vars)

        self.G_activation_loss = self.G_loss + .000 * self.activation_loss

        # Create Optimizers
        self.opti_D = tf.train.AdamOptimizer(learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        self.opti_G = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.G_activation_loss, var_list=self.g_vars)

        self.combined_loss = 1 * self.activation_loss + 1 * self.basic_loss

        self.combined_optimizer = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.combined_loss, var_list=self.g_vars)

        self.basic_optimizer = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.basic_loss, var_list=self.g_vars)

        self.activation_optimizer = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.activation_loss, var_list=self.g_vars)

    def load_reference_model(self, input_model_path=None):

        # from deepneuro.local.basic_cnn import cnn_baseline
        # model = cnn_baseline()
        # model.load_weights('../Interp_GAN/classification_model/cnn/best_cnn_weights_5Classes_RGB_Flip_Rot_0.001.hdf5')

        load_old_model('DRIVE_segmentation_unet.h5')

        return

    def load_model(self, input_model_path, batch_size=1):

        self.build_tensorflow_model(batch_size)
        self.init_sess()
        self.saver.restore(self.sess, os.path.join(input_model_path, 'model.ckpt'))

    def predict(self, sample_latent=None, batch_size=1):

        self.init_sess()

        if sample_latent is None:
            sample_latent = np.random.normal(size=[batch_size, self.latent_size])

        return self.sess.run(self.synthetic_images, feed_dict={self.latent: sample_latent})