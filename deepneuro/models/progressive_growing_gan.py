"""This is a vanilla implementation of a generative adversarial network. It includes
the Wasserstein Gradient-Penalty by default.
"""

import math
import numpy as np
import tensorflow as tf
import os
import imageio
import keras
import scipy

from tqdm import tqdm
from collections import defaultdict

from deepneuro.utilities.util import add_parameter
from deepneuro.models.blocks import generator, discriminator
from deepneuro.models.cost_functions import wasserstein_loss
from deepneuro.models.gan import GAN
from deepneuro.models.ops import upscale2d, downscale2d
from deepneuro.utilities.visualize import check_data


class PGGAN(GAN):
    
    def load(self, kwargs):

        """ Parameters
            ----------
            initial_size : tuple


        """

        super(PGGAN, self).load(kwargs)

        add_parameter(self, kwargs, 'initial_size', (4, 4))

        # Dummy training variables.
        add_parameter(self, kwargs, 'num_epochs', 1)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 1)
        add_parameter(self, kwargs, 'training_batch_size', 16)
        add_parameter(self, kwargs, 'batch_size', None)

        if self.batch_size is not None:
            self.training_batch_size = self.batch_size

        if type(self.initial_size) not in [list, tuple]:
            self.initial_size = (4,) * self.dim

        # PGGAN Parameters
        self.starting_depth = 1
        self.transition_dict = {True: '_Transition', False: ''}
        self.progressive_depth = self.depth
        self.transition = False

        if self.dim == 3:
            raise NotImplementedError 

    def get_filter_num(self, depth):

        # This will need to be a bit more complicated; see PGGAN paper.
        if self.max_filter / (2 ** (depth)) <= self.filter_floor:
            return self.filter_floor
        else:
            return min(self.max_filter / (2 ** (depth)), self.filter_cap)

    def init_training(self, training_data_collection, kwargs):

        # Outputs
        add_parameter(self, kwargs, 'output_model_filepath')

        # Training Parameters
        add_parameter(self, kwargs, 'num_epochs', 100)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 10)
        add_parameter(self, kwargs, 'training_batch_size', 16)
        add_parameter(self, kwargs, 'callbacks')

        self.callbacks = self.get_callbacks(backend='tensorflow', model=self, batch_size=self.training_batch_size, **kwargs)

        self.create_data_generators(training_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)

        return

    def train(self, training_data_collection, validation_data_collection=None, **kwargs):

        """ TODO: Make the model naming structure more explicit.
            TODO: Create more explicit documentation for callbacks.
        """

        self.init_training(training_data_collection, kwargs)

        if not os.path.exists(self.output_model_filepath):
            os.mkdir(self.output_model_filepath)

        self.callback_process('on_train_begin')

        # Some explanation on training stages: The progressive gan trains each resolution
        # in two stages. One interpolates from the previous resolution, while one trains 
        # solely on the current resolution. The loop below looks odd because the lowest 
        # resolution only has one stage.

        for self.training_stage in range(int(np.ceil((self.starting_depth - 1) / 2)), (self.depth * 2) - 1):

            if (self.training_stage % 2 == 0):
                self.transition = False
            else:
                self.transition = True

            current_depth = np.ceil((self.training_stage + 2) / 2)
            previous_depth = np.ceil((self.training_stage + 1) / 2)
            self.progressive_depth = int(current_depth)

            current_model_path = os.path.join(self.output_model_filepath, '{}{}'.format(str(current_depth), self.transition_dict[self.transition]), 'model.ckpt')
            if not os.path.exists(os.path.dirname(current_model_path)):
                os.mkdir(os.path.dirname(current_model_path))

            previous_model_path = os.path.join(self.output_model_filepath, '{}{}'.format(str(previous_depth), self.transition_dict[not self.transition]), 'model.ckpt')

            self.callback_process('on_depth_begin', [current_depth, self.transition])

            self.init_sess()
            self.build_tensorflow_model(self.training_batch_size)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

            if self.transition:
                self.r_saver.restore(self.sess, previous_model_path)
                self.rgb_saver.restore(self.sess, previous_model_path)
            elif self.training_stage > int(np.ceil((self.starting_depth - 1) / 2)):
                self.saver.restore(self.sess, previous_model_path)

            for epoch in range(self.num_epochs):

                print(('Depth {}/{}, Epoch {}/{}'.format(self.progressive_depth, self.depth, epoch + 1, self.num_epochs)))
                self.callback_process('on_epoch_begin', '_'.join([str(current_depth), str(epoch)]))

                step_counter = tqdm(list(range(self.training_steps_per_epoch)), total=self.training_steps_per_epoch, unit="step", desc="Generator Loss:", miniters=1)

                for step in step_counter:

                    self.callback_process('on_batch_begin', step)

                    reference_data = self.process_step(step_counter, step, epoch)

                    self.callback_process('on_batch_end', step)

                self.callback_process('on_epoch_end', [str(epoch), reference_data])

                self.saver.save(self.sess, current_model_path)

            self.callback_process('on_depth_end', [current_depth, self.transition])

            self.saver.save(self.sess, current_model_path)

            self.sess.close()
            tf.reset_default_graph()

        # Should this be called after each progression, or after all training?
        self.callback_process('on_train_end')

        return

    # @profile
    def process_step(self, step_counter, step, epoch):

        for i in range(self.discriminator_updates):

            sample_latent = np.random.normal(size=[self.training_batch_size, self.latent_size])

            reference_data = next(self.training_data_generator)[self.input_data]
            reference_data = self.sess.run(self.input_volumes, feed_dict={self.raw_volumes: reference_data})

            if self.transition:
                reference_data = self.sess.run(self.real_images, feed_dict={self.reference_images: reference_data})

            self.sess.run(self.opti_D, feed_dict={self.reference_images: reference_data, self.latent: sample_latent})

        # Update Generator
        for i in range(self.generator_updates):

            self.sess.run(self.opti_G, feed_dict={self.latent: sample_latent})

        # the alpha of fake_in process
        if self.transition:
            self.sess.run(self.alpha_transition_assign, feed_dict={self.step_pl: (epoch * float(self.training_steps_per_epoch)) + (step + 1)})

        d_loss, d_loss_origin, g_loss, transition = self.sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_transition], feed_dict={self.reference_images: reference_data, self.latent: sample_latent})

        self.log([d_loss, d_loss_origin, g_loss, transition], headers=['Dis-WP Loss', 'Dis Loss', 'Gen Loss', 'Alpha'], verbose=self.hyperverbose)
        
        if self.transition:
            step_counter.set_description("Generator Loss: {0:.5f}".format(g_loss) + " Discriminator Loss: {0:.5f}".format(d_loss) + " Alpha: {0:.2f}".format(transition))
        else:
            step_counter.set_description("Generator Loss: {0:.5f}".format(g_loss) + " Discriminator Loss: {0:.5f}".format(d_loss))

        summary_str = self.sess.run(self.summary_op, feed_dict={self.reference_images: reference_data, self.latent: sample_latent})
        if self.tensorboard_directory is not None:
            self.summary_writer.add_summary(summary_str, (self.num_epochs * self.training_steps_per_epoch * self.training_stage) + (self.training_steps_per_epoch * epoch) + step)

        return reference_data

    def build_tensorflow_model(self, batch_size):

        """ TODO: Break out into functions?
            TODO: Create progressive growing for non-square and non-2-power resolutions.
            TODO: Investigate cause of loss jumps at transitions.
        """

        # Set input/output shapes for reference during inference.
        self.model_input_shape = tuple([batch_size] + list(self.input_shape))
        self.model_output_shape = tuple([batch_size] + list(self.input_shape))

        self.alpha_transition = tf.Variable(initial_value=0.0, trainable=False, name='alpha_transition')
        self.step_pl = tf.placeholder(tf.float32, shape=None)
        self.alpha_transition_assign = self.alpha_transition.assign(self.step_pl / (self.num_epochs * self.training_steps_per_epoch))

        self.latent = tf.placeholder(tf.float32, [None, self.latent_size])
        self.reference_images = tf.placeholder(tf.float32, [None] + list(self.model_input_shape)[1:])
        self.synthetic_images = generator(self, self.latent, depth=self.progressive_depth - 1, transition=self.transition, alpha_transition=self.alpha_transition, name='generator')

        # Derived Parameters
        self.output_size = pow(2, self.progressive_depth + 1)
        self.zoom_level = self.progressive_depth
        self.reference_images = tf.placeholder(tf.float32, [None] + [self.output_size] * self.dim + [self.channels])

        max_downscale = np.floor(math.log(self.model_input_shape[1], 2))
        downscale_factor = 2 ** max_downscale / (2 ** (self.progressive_depth + 1))
        self.raw_volumes = tf.placeholder(tf.float32, self.model_input_shape)
        self.input_volumes = downscale2d(self.raw_volumes, downscale_factor)
        # Data Loading Tools
        self.low_images = upscale2d(downscale2d(self.reference_images, 2), 2)
        self.real_images = self.alpha_transition * self.reference_images + (1 - self.alpha_transition) * self.low_images

        self.discriminator_real, self.discriminator_real_logits = discriminator(self, self.reference_images, depth=self.progressive_depth - 1, name='discriminator', transition=self.transition, alpha_transition=self.alpha_transition)
        self.discriminator_fake, self.discriminator_fake_logits = discriminator(self, self.synthetic_images, depth=self.progressive_depth - 1, name='discriminator', transition=self.transition, alpha_transition=self.alpha_transition, reuse=True)

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

        self.calculate_losses()
        self.log_variables()

        if self.hyperverbose:
            self.model_summary()

    def calculate_losses(self):

        self.D_loss, self.G_loss, self.D_origin_loss = wasserstein_loss(self, discriminator, self.discriminator_fake_logits, self.discriminator_real_logits, self.synthetic_images, self.real_images, gradient_penalty_weight=self.gradient_penalty_weight, name='discriminator', depth=self.progressive_depth - 1, transition=self.transition, alpha_transition=self.alpha_transition, dim=self.dim)

        # A little sketchy. Attempting to make variable loss functions extensible later.
        self.D_loss = self.D_loss[0]
        self.G_loss = self.G_loss[0]
        self.D_origin_loss = self.D_origin_loss[0]

        # Create Optimizers
        self.opti_D = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        self.opti_G = self.tensorflow_optimizer_dict[self.optimizer](learning_rate=self.initial_learning_rate, beta1=0.0, beta2=0.99).minimize(self.G_loss, var_list=self.g_vars)

    def log_variables(self):

        tf.summary.scalar('Generator Loss', self.G_loss)
        tf.summary.scalar('Discriminator Loss (WP)', self.D_loss)
        tf.summary.scalar('Discriminator Loss (Basic)', self.D_origin_loss)
        tf.summary.scalar('Interpolation %', self.alpha_transition)

        super(PGGAN, self).log_variables()

        return

    def load_model(self, input_model_path, batch_size=1):

        self.init_sess()
        self.build_tensorflow_model(batch_size)
        model = os.path.join(input_model_path, '{}'.format(str(float(self.depth))), 'model.ckpt')
        self.saver.restore(self.sess, model)

    def get_callbacks(self, callbacks=[], output_model_filepath=None, monitor='val_loss', model=None, data_collection=None, save_best_only=False, epoch_prediction_dir=None, batch_size=1, epoch_prediction_object=None, epoch_prediction_data_collection=None, epoch_prediction_batch_size=None, latent_size=128, backend='tensorflow', **kwargs):

        """ Very disorganized currently. Replace with dictionary? Also address never-ending parameters
        """

        return_callbacks = []

        for callback in callbacks:

            if callback == 'predict_gan':
                return_callbacks += [PGGANPredict(deepneuro_model=model, data_collection=data_collection, epoch_prediction_dir=epoch_prediction_dir, batch_size=batch_size, epoch_prediction_object=epoch_prediction_object, epoch_prediction_data_collection=epoch_prediction_data_collection, epoch_prediction_batch_size=epoch_prediction_batch_size, latent_size=latent_size)]

        return return_callbacks


class PGGANPredict(keras.callbacks.Callback):

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'epoch_prediction_data_collection', self.data_collection)
        add_parameter(self, kwargs, 'epoch_prediction_object', None)
        add_parameter(self, kwargs, 'deepneuro_model', None)
        add_parameter(self, kwargs, 'epoch_prediction_dir', None)
        add_parameter(self, kwargs, 'output_gif', None)
        add_parameter(self, kwargs, 'batch_size', 1)
        add_parameter(self, kwargs, 'epoch_prediction_batch_size', self.batch_size)
        add_parameter(self, kwargs, 'latent_size', 128)
        add_parameter(self, kwargs, 'sample_latent', None)

        if self.sample_latent is None:
            self.sample_latent = np.random.normal(size=[self.epoch_prediction_batch_size, self.latent_size])

        if not os.path.exists(self.epoch_prediction_dir):
            os.mkdir(self.epoch_prediction_dir)

        self.predictions = []
        self.depth_directories = []

    def on_train_end(self, logs={}):
        
        """ Very confusing, make more explicit
        """

        for key in list(self.predictions[-1].keys()):

            max_size = self.predictions[-1][key][0].shape[0]
            final_predictions = []

            for predictions in self.predictions:

                new_predictions = []
                data = predictions[key]

                if data[0].shape[0] != max_size:
                    upsample_ratio = max_size / data[0].shape[0]
                    for prediction in data:
                        if prediction.shape[-1] == 3:
                            new_predictions += [scipy.misc.imresize(prediction, upsample_ratio * 100, interp='nearest')]
                        elif prediction.shape[-1] == 1:
                            new_predictions += [scipy.misc.imresize(np.repeat(prediction, 3, axis=2), upsample_ratio * 100, interp='nearest')]
                else:
                    new_predictions = data

                final_predictions += new_predictions

            imageio.mimsave(os.path.join(self.epoch_prediction_dir, 'pggan_training_' + key + '.gif'), final_predictions)

        return
 
    def on_epoch_end(self, data, logs={}):

        # Hacky, revise later.
        epoch = data[0]
        reference_data = data[1][0:self.epoch_prediction_batch_size]

        if self.epoch_prediction_object is None:
            prediction = self.deepneuro_model.predict(sample_latent=self.sample_latent)
        else:
            prediction = self.epoch_prediction_object.process_case(self.predict_data[self.deepneuro_model.input_data], model=self.deepneuro_model)

        output_filepaths, output_images = check_data({'prediction': prediction, 'real_data': reference_data}, output_filepath=os.path.join(self.depth_dir, 'epoch_{}.png'.format(epoch)), show_output=False, batch_size=self.epoch_prediction_batch_size)

        for key, images in output_images.items():
            self.predictions[-1][key] += [images.astype('uint8')]

        return

    def on_depth_begin(self, depth_transition, logs={}):

        self.depth_dir = os.path.join(self.epoch_prediction_dir, '{}_{}'.format(str(depth_transition[0]), str(depth_transition[1])))

        if not os.path.exists(self.depth_dir):
            os.mkdir(self.depth_dir)

        self.predictions += [defaultdict(list)]

        return

    def on_depth_end(self, depth_transition, logs={}):

        for key, images in self.predictions[-1].items():
            imageio.mimsave(os.path.join(self.depth_dir, 'epoch_prediction_' + key + '.gif'), images)

        return