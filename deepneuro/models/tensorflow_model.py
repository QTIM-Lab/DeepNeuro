import os
import glob
import tensorflow as tf
import numpy as np

from shutil import rmtree
from tqdm import tqdm

from deepneuro.models.model import DeepNeuroModel
from deepneuro.utilities.util import add_parameter


class TensorFlowModel(DeepNeuroModel):

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

        add_parameter(self, kwargs, 'sess', None)
        add_parameter(self, kwargs, 'saver', None)

        self.tensorflow_optimizer_dict = {'Adam': tf.train.AdamOptimizer}

    def init_training(self, training_data_collection, kwargs):

        # Outputs
        add_parameter(self, kwargs, 'output_model_filepath')

        # Training Parameters
        add_parameter(self, kwargs, 'num_epochs', 100)
        add_parameter(self, kwargs, 'training_steps_per_epoch', 10)
        add_parameter(self, kwargs, 'training_batch_size', 16)

        self.init_sess()
        self.build_tensorflow_model(self.training_batch_size)
        self.create_data_generators(training_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)

        return

    def train(self, training_data_collection, validation_data_collection=None, **kwargs):

        self.init_training(training_data_collection, kwargs)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.callback_process('on_train_begin')

        try:

            for epoch in range(self.num_epochs):

                print(('Epoch {}/{}'.format(epoch, self.num_epochs)))
                self.callback_process('on_epoch_begin', epoch)

                step_counter = tqdm(list(range(self.training_steps_per_epoch)), total=self.training_steps_per_epoch, unit="step", desc="Generator Loss:", miniters=1)

                for step in step_counter:

                    self.callback_process('on_batch_begin', step)

                    self.process_step(step_counter)

                    self.callback_process('on_batch_end', step)

                self.callback_process('on_epoch_end', epoch)

            self.callback_process('on_train_end')

        except KeyboardInterrupt:

            self.callback_process('on_train_end')
        except:
            raise

    def process_step(self):

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

    def init_sess(self):

        if self.sess is None:
            self.graph = tf.Graph()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.InteractiveSession(config=config, graph=self.graph)

        elif self.sess._closed:
            self.graph = tf.Graph()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.InteractiveSession(config=config, graph=self.graph)

    def save_model(self, output_model_filepath, overwrite=True):

        self.init_sess()

        if output_model_filepath.endswith(('.h5', '.hdf5')):
            output_model_filepath = '.'.join(str.split(output_model_filepath, '.')[0:-1])

        if os.path.exists(output_model_filepath) and overwrite:
            rmtree(output_model_filepath)

        if self.saver is None:
            self.saver = tf.train.Saver()

        save_path = self.saver.save(self.sess, os.path.join(output_model_filepath, "model.ckpt"))

        return save_path

    def log_variables(self):
        self.summary_op = tf.summary.merge_all()
        if self.tensorboard_directory is not None:
            if self.tensorboard_run_directory is None:
                previous_runs = glob.glob(os.path.join(self.tensorboard_directory, 'tensorboard_run*'))
                if len(previous_runs) == 0:
                    run_number = 0
                else:
                    run_number = max([int(s.split('tensorboard_run_')[1]) for s in previous_runs]) + 1
                self.tensorboard_run_directory = os.path.join(self.tensorboard_directory, 'tensorboard_run_%02d' % run_number)
            self.summary_writer = tf.summary.FileWriter(self.tensorboard_run_directory, self.sess.graph)

    def model_summary(self):

        for layer in tf.trainable_variables():
            print(layer)

    def callback_process(self, command='', idx=None):

        for callback in self.callbacks:
            if type(callback) is str:
                continue
            method = getattr(callback, command)
            method(idx)

        return

    def grab_tensor(self, layer):
        return self.graph.get_tensor_by_name(layer + ':0')

    def find_layers(self, contains=['discriminator/']):

        for layer in self.graph.get_operations():
            if any(op_type in layer.name for op_type in contains):
                try:
                    if self.graph.get_tensor_by_name(layer.name + ':0').get_shape() != ():
                        print((layer.name, self.graph.get_tensor_by_name(layer.name + ':0').get_shape()))
                except:
                    continue

    def load_model(self, input_model_path, batch_size=1):

        self.build_tensorflow_model(batch_size)
        self.init_sess()
        self.saver.restore(self.sess, os.path.join(input_model_path, 'model.ckpt'))