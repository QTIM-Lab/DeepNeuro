""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

import tensorflow as tf
import os

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

        # Discriminator Parameters
        add_parameter(self, kwargs, 'discriminator_depth', 4)
        add_parameter(self, kwargs, 'discriminator_max_filter', 4096)

        # GAN Parameters
        add_parameter(self, kwargs, 'latent_size', 128)

        # PGGAN Parameters
        add_parameter(self, kwargs, 'progressive_depth', 9)
        add_parameter(self, kwargs, 'transition', True)
        add_parameter(self, kwargs, 'filter_cap', 128)
        add_parameter(self, kwargs, 'filter_floor', 16)

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

    def train(self, training_data_collection, validation_data_collection=None, **kwargs):

        # Outputs
        add_parameter(self, kwargs, 'output_model_filepath', 'pgan_model')

        # Training Parameters
        add_parameter(self, kwargs, 'num_epochs', 100)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 10)
        add_parameter(self, kwargs, 'training_batch_size', 16)

        self.build_tensorflow_model(self.training_batch_size)
        self.create_data_generators(training_data_collection, validation_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)
        self.init_sess()

        # step = 0

        # for epoch in range(self.num_epochs):

        #     for step in range(self.training_steps_per_epoch):
                
        #         print epoch, step
        #         input_modality_1, input_modality_2 = next(self.training_data_generator)

        #         # Optimize!

        #         if self.train_with_GAN:

        #             _, _, discrim_1_loss, discrim_2_loss, d_loss, generator_1_loss, generator_2_loss, cons_1_loss, cons_2_loss, g_loss = self.sess.run([self.generator_optimizer, self.discriminator_optimizer, self.D_loss_wgan_2, self.D_loss_wgan_1, self.total_D_loss, self.G_loss_1_2, self.G_loss_2_1, self.generator_1_consistency_loss, self.generator_2_consistency_loss, self.total_G_loss], feed_dict={self.generator_input_images_1: input_modality_1, self.generator_input_images_2: input_modality_2})

        #             self.log([discrim_1_loss, discrim_2_loss, d_loss, generator_1_loss, generator_2_loss, cons_1_loss, cons_2_loss, g_loss], headers=['Dis 1 Loss', 'Dis 2 Loss', 'Total D Loss', 'Gen 1 Loss', 'Gen 2 Loss', 'Consistency 12 Loss', 'Consistency 21 Loss', 'Total G Loss'], verbose=self.hyperverbose)

        #         else:

        #             _, cons_1_loss, cons_2_loss, g_loss = self.sess.run([self.consistency_optimizer, self.generator_2_consistency_loss, self.generator_1_consistency_loss, self.total_consistency_loss], feed_dict={self.generator_input_images_1: input_modality_1, self.generator_input_images_2: input_modality_2})

        #             self.log([cons_1_loss, cons_2_loss, g_loss], headers=['Consistency Loss 12', 'Consistency Loss 21', 'Total G Loss'], verbose=self.hyperverbose)

        #     self.save_model(self.output_model_filepath)


        # Create fade-in (transition) parameters.
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_transition_assign = self.alpha_transition.assign(step_pl / self.max_iterations)

        # Create Optimizers
        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # Personally have no idea what is being logged in this thing --andrew
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            # No idea what the saving systems is like. TODO investigate --andrew.
            # I don't think you need to save and reload models if you create a crazy
            # system where you're only optimizing certain outputs/cost functions at
            # any one time.
            if self.progressive_depth != 1:

                if self.transition:
                    self.r_saver.restore(sess, self.input_model_path)
                    self.rgb_saver.restore(sess, self.input_model_path)
                else:
                    self.saver.restore(sess, self.input_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iterations:

                n_critic = 1

                # Update Discriminator
                for i in range(n_critic):

                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])

                    if self.classes:
                        realbatch_array, _ = self.training_data.get_next_batch_classed(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)
                    else:
                        realbatch_array, _ = self.training_data.get_next_batch(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)

                    realbatch_array = realbatch_array[...,0:4]

                    if self.transition:
                        
                        realbatch_array = sess.run(self.real_images, feed_dict={self.images: realbatch_array})

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    batch_num += 1

                # Update Generator
                sess.run(opti_G, feed_dict={self.latent: sample_latent})

                summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                summary_writer.add_summary(summary_str, step)

                # the alpha of fake_in process
                sess.run(alpha_transition_assign, feed_dict={step_pl: step})

                if step % 40 == 0:
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_transition], feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.progressive_depth, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                if step % 400 == 0:

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_real.png'.format(self.samples_dir, step))

                    fake_image = sess.run(self.fake_images, feed_dict={self.images: realbatch_array, self.latent: sample_latent})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.png'.format(self.samples_dir, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.output_model_path)
                step += 1

            save_path = self.saver.save(sess, self.output_model_path)
            print "Model saved in file: %s" % save_path

        tf.reset_default_graph()

        return

    def build_tensorflow_model(self, batch_size):

        # Set input/output shapes for reference during inference.
        self.model_input_shape = tuple([batch_size] + list(self.input_shape))
        self.model_output_shape = tuple([batch_size] + list(self.input_shape))

        self.latent = tf.placeholder(tf.float32, [self.training_batch_size, self.latent_size])

        # Derived Parameters
        self.output_size = pow(2, self.progressive_depth + 1)
        self.zoom_level = self.progressive_depth
        self.images = tf.placeholder(tf.float32, [self.training_batch_size, self.output_size, self.output_size, self.channels])
        # self.seg_images = tf.placeholder(tf.float32, [self.training_batch_size, self.output_size, self.output_size, 1])
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

        for layer in self.d_vars + self.g_vars:
            print layer

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def discriminator(self, input_image, reuse=False, name=None, progressive_depth=1, transition=False, alpha_transition=0.01):

        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            if transition:
                transition_conv = DnAveragePooling(input_image, dim=self.dim)
                transition_conv = leaky_relu(DnConv(transition_conv, output_dim=self.get_filter_num(progressive_depth - 2, self.discriminator_max_filter), kernel_size=(1, 1), stride_size=(1, 1), name='discriminator_y_rgb_conv_{}'.format(transition_conv.shape[1]), dim=self.dim))

            convs = []

            # fromRGB
            convs += [leaky_relu(DnConv(input_image, output_dim=self.get_filter_num(progressive_depth - 1, self.discriminator_max_filter), kernel_size=(1, 1), stride_size=(1, 1), name='discriminator_y_rgb_conv_{}'.format(input_image.shape[1]), dim=self.dim))]

            for i in range(progressive_depth - 1):

                convs += [leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(progressive_depth - 1 - i, self.discriminator_max_filter), stride_size=(1, 1), name='discriminator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=self.dim))]

                convs += [leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(progressive_depth - 2 - i, self.discriminator_max_filter), stride_size=(1, 1), name='discriminator_n_conv_2_{}'.format(convs[-1].shape[1]), dim=self.dim))]
                convs[-1] = DnAveragePooling(convs[-1], dim=self.dim)

                if i == 0 and transition:
                    convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

            convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(1, self.discriminator_max_filter), kernel_size=(3, 3), stride_size=(1, 1), name='discriminator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=self.dim))
            
            if False:
                convs[-1] = leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(1, self.discriminator_max_filter), kernel_size=(4, 4), stride_size=(1, 1), padding='VALID', name='discriminator_n_conv_2_{}'.format(convs[-1].shape[1]), dim=self.dim))

            for conv in convs:
                print conv
        
            #for D
            output = tf.reshape(convs[-1], [self.training_batch_size, -1])

            if self.classify is None:
                discriminate_output = dense(output, output_size=1, scope='discriminator_n_fully')
                return None, None, tf.nn.sigmoid(discriminate_output), discriminate_output

    def generator(self, latent_var, progressive_depth=1, name=None, transition=False, alpha_transition=0.0):

        with tf.variable_scope(name) as scope:

            convs = []

            convs += [tf.reshape(latent_var, [self.training_batch_size, 1, 1, self.latent_size])]

            convs[-1] = pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(1, self.generator_max_filter), kernel_size=(4, 4), stride_size=(1, 1), padding='Other', name='generator_n_1_conv', dim=self.dim)))

            convs += [tf.reshape(convs[-1], [self.training_batch_size, 4, 4, self.get_filter_num(1, self.generator_max_filter)])] # why necessary? --andrew
            convs[-1] = pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(1, self.generator_max_filter), stride_size=(1, 1), name='generator_n_2_conv', dim=self.dim)))

            for i in range(progressive_depth - 1):

                if i == progressive_depth - 2 and transition:  # redundant conditions? --andrew
                    #To RGB
                    # Don't totally understand this yet, diagram out --andrew
                    transition_conv = DnConv(convs[-1], output_dim=self.channels, kernel_size=(1, 1), stride_size=(1, 1), name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=self.dim)
                    transition_conv = upscale(transition_conv, 2)

                convs += [upscale(convs[-1], 2)]
                convs[-1] = pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(i + 1, self.generator_max_filter), stride_size=(1, 1), name='generator_n_conv_1_{}'.format(convs[-1].shape[1]), dim=self.dim)))

                convs += [pixel_norm(leaky_relu(DnConv(convs[-1], output_dim=self.get_filter_num(i + 1, self.generator_max_filter), stride_size=(1, 1), name='generator_n_conv_2_{}'.format(convs[-1].shape[1]), dim=self.dim)))]

            #To RGB
            convs += [DnConv(convs[-1], output_dim=self.channels, kernel_size=(1, 1), stride_size=(1, 1), name='generator_y_rgb_conv_{}'.format(convs[-1].shape[1]), dim=self.dim)]

            if transition:
                convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

            for conv in convs:
                print conv

            return convs[-1]

    def load_model(self, input_model_path, batch_size=1):

        self.build_tensorflow_model(batch_size)
        self.init_sess()
        self.saver.restore(self.sess, os.path.join(input_model_path, 'model.ckpt'))

    def predict(self, input_data):

        self.init_sess()

        return self.sess.run(self.generator_1_2_real, feed_dict={self.generator_input_images_1: input_data})

    def wasserstein_loss(self, discriminator_loss, real_data, fake_data, batch_size, gradient_penalty_weight=10, name=''):

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
        discriminator_loss += gradient_penalty_weight * gradient_penalty
        # discriminator_loss += 0.001 * tf.reduce_mean(tf.square(self.discriminator_2_real_logits - 0.0))

        return discriminator_loss

    def discriminator(self, input_image, reuse=False, name=None, scope=None, transition=False, alpha_transition=0.01):

        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            convs = []

            convs += [lrelu(DnConv(input_image, output_dim=self.discriminator_max_filter / self.discriminator_depth, kernel_size=(1, 1, 1), name='dis_y_rgb_conv_{}'.format(input_image.shape[1])))]

            for i in range(self.discriminator_depth - 1):

                convs += [lrelu(DnConv(convs[-1], output_dim=self.discriminator_max_filter / (self.discriminator_depth - i - 1), stride_size=(1, 1, 1), name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))]

                convs += [lrelu(DnConv(convs[-1], output_dim=self.discriminator_max_filter / (self.discriminator_depth - 1 - i), stride_size=(1, 1, 1), name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))]
                convs[-1] = DnAveragePooling(convs[-1], ratio=(2, 2, 2))

            # convs += [minibatch_state_concat(convs[-1])] 
            convs[-1] = lrelu(DnConv(convs[-1], output_dim=self.discriminator_max_filter, kernel_size=(3, 3, 3), stride_size=(1, 1, 1), name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            # conv = lrelu(DnConv(convs[-1], output_dim=self.discriminator_max_filter, kernel_size=(2, 2, 2), stride_size=(1, 1, 1), padding='VALID', name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))
            
            #for D
            output = tf.layers.Flatten()(convs[-1])
            output = fully_connect(output, output_size=1, scope='dis_n_fully')
            # fd = dg

            return tf.nn.sigmoid(output), output