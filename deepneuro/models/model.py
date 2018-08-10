""" model.py specified the basic DeepNeuroModel class, as well as some model utilites and
    pre-built model chunks that may be useful across all models.
"""

import os
import tensorflow as tf
import csv

from shutil import rmtree
from tqdm import tqdm
from keras.engine import Input
from keras.models import load_model

from deepneuro.models.cost_functions import cost_function_dict
from deepneuro.utilities.util import add_parameter
from deepneuro.utilities.visualize import check_data
from deepneuro.models.callbacks import get_callbacks


class DeepNeuroModel(object):
    
    def __init__(self, model=None, downsize_filters_factor=1, pool_size=(2, 2, 2), filter_shape=(3, 3, 3), dropout=.1, batch_norm=False, initial_learning_rate=0.00001, output_type='regression', num_outputs=1, padding='same', implementation='keras', **kwargs):

        """A model object with some basic parameters that can be added to in the load() method. Each child of
        this class should be able to build and store a model composed of tensors, as well as convert an input
        tensor into an output tensor according to a model's schema. 
        
        TODO: Add a parameter for optimizer type.
        TODO: Add option for outputting multiple labels.
        
        Parameters
        ----------
        input_shape : tuple, optional
            Input dimensions of first layer. Not counting batch-size.
        input_tensor : tensor, optional
            If input_tensor is specified, build_model will output a tensor
            created from input_tensor.
        downsize_filters_factor : int, optional
            If specified, the number of filters at each applicable layer in a model
            will be divided by downsize_filters_factor (with rounding if necessary).
            Different models may specify different implementations of the downsizing
            process.
        pool_size : tuple, optional
            Pool size for convolutional layers.
        filter_shape : tuple, optional
            Filter size for convolutional layers.
        dropout : float, optional
            Dropout percentage for children models. Each model's implementation of
            dropout will differ.
        batch_norm : bool, optional
            Whether layers are batch-normed in children models. Each model's implementation
            of which layers will be batch-normed is different.
        initial_learning_rate : float, optional
            Initial learning rate for the chosen optimizer type, if necessary.
        output_type : str, optional
            Currently can choose from 'regression', which is an output the same size as input,
            'binary_label' which is a binary probability map, or 'categorical_label'
        num_outputs : int, optional
            If output_type is 'categorical_label', this specified how many outputs.
        activation : str, optional
            What type of activation to use at each layer. May be implemented differently in
            each model.
        padding : str, optional
            Padding for convolutional layers.
        implementation : str, optional
            Determines whether this is a 'tensorflow' or 'keras' model at present. [PROVISIONAL]
        **kwargs
            Addtional variables that may be needed in children classes.
        """

        # Inputs
        add_parameter(self, kwargs, 'input_shape', (32, 32, 32, 1))
        add_parameter(self, kwargs, 'input_tensor', None)
        add_parameter(self, kwargs, 'dim', len(self.input_shape) - 1)
        add_parameter(self, kwargs, 'channels', self.input_shape[-1])

        if self.input_tensor is None:
            self.inputs = Input(self.input_shape)
        else:
            self.inputs = self.input_tensor

        # Generic Model Parameters -- Optional
        self.pool_size = pool_size
        self.filter_shape = filter_shape
        self.padding = padding
        self.downsize_filters_factor = downsize_filters_factor
        add_parameter(self, kwargs, 'max_filter', 512)
        add_parameter(self, kwargs, 'kernel_size', (3, 3, 3))
        add_parameter(self, kwargs, 'stride_size', (1, 1, 1))
        add_parameter(self, kwargs, 'activation', 'relu')
        add_parameter(self, kwargs, 'optimizer', 'Adam')
        add_parameter(self, kwargs, 'cost_function', 'mean_squared_error')

        self.dropout = dropout
        self.batch_norm = batch_norm

        self.initial_learning_rate = initial_learning_rate

        # Logging Parameters - Temporary
        add_parameter(self, kwargs, 'output_log_file', 'deepneuro_log.csv')

        # DeepNeuro Parameters
        add_parameter(self, kwargs, 'input_data', 'input_data')
        add_parameter(self, kwargs, 'targets', 'ground_truth')

        # Misc
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'hyperverbose', False)

        # Derived Parameters
        self.write_file = None
        self.csv_writer = None

        self.num_outputs = num_outputs
        self.output_type = output_type

        self.implementation = implementation

        self.load(kwargs)

        self.model = model

        if self.model is None:
            self.build_model()

        self.outputs = []

        pass

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def build_model(self):

        """ This method is inherited by child classes to specify the classes model attribute. If input_tensor
            is specified, build_model returns a tensor output. If not, it return a Keras model output.
        """

        return None

    def model(self):

        """ Access this class's model attribute. TODO: investigate overloading.
        """

        if self.model is None:
            self.build_model()

        return self.model

    def train(self):

        return

    def append_output(self, outputs):

        for output in outputs:
            self.outputs += [output]

    def clear_outputs(self):

        self.outputs = []

    def generate_outputs(self, data_collection, case=None):

        callbacks = []

        for output in self.outputs:
            # A little odd.
            output.model, output.data_collection, output.case = self, data_collection, case
            callbacks += [output.generate()]

        return callbacks

    def create_data_generators(self, training_data_collection, validation_data_collection=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None):

        if training_steps_per_epoch is None:
            self.training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1
        else:
            self.training_steps_per_epoch = training_steps_per_epoch

        self.training_data_generator = training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=training_batch_size)

        if validation_data_collection is not None:

            if validation_steps_per_epoch is None:
                self.validation_steps_per_epoch = validation_data_collection.total_cases // validation_batch_size + 1
            else:
                self.validation_steps_per_epoch = validation_steps_per_epoch

            self.validation_data_generator = validation_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=validation_batch_size)

        else:

            self.validation_data_generator = None

    def predict(self, input_data):

        return

    def log(self, inputs=None, headers=None, verbose=False):

        if self.write_file is None:
            self.write_file = open(self.output_log_file, 'w')
            self.csv_writer = csv.writer(self.write_file)
            if headers is not None:
                self.csv_writer.writerow(headers)

        if inputs is not None:
            self.csv_writer.writerow(inputs)

        if verbose:
            for input_idx, single_input in enumerate(inputs):
                if headers is None:
                    print('Logging Output', input_idx, single_input)
                else:
                    print(headers[input_idx], single_input)

        return

    def close_model(self):

        if self.write_file is not None:
            if self.write_file.open:
                self.write_file.close()

    def fit_one_batch(self, training_data_collection, output, output_directory):

        return


class KerasModel(DeepNeuroModel):

    def load(self, kwargs):

        super(KerasModel, self).load(kwargs)

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model', 'log'], **kwargs):

        """
        input_groups : list of strings, optional
            Specifies which named data groups (e.g. "ground_truth") enter which input
            data slot in your model.
        """

        # Todo: investigate call-backs more thoroughly.
        # Also, maybe something more general for the difference between training and validation.
        # Todo: list-checking for callbacks

        self.create_data_generators(training_data_collection, validation_data_collection, input_groups, training_batch_size, validation_batch_size, training_steps_per_epoch, validation_steps_per_epoch)

        if validation_data_collection is None:
            self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, callbacks=get_callbacks(callbacks=callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, batch_size=training_batch_size, model=self, backend='keras', **kwargs))

        else:
            self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, validation_data=self.validation_data_generator, validation_steps=self.validation_steps_per_epoch, callbacks=get_callbacks(callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, model=self, batch_size=training_batch_size, backend='keras', **kwargs))

        return

    def fit_one_batch(self, training_data_collection, output_model_filepath=None, input_groups=None, output_directory=None, callbacks=['save_model', 'log'], training_batch_size=16, training_steps_per_epoch=None, num_epochs=None, show_results=True, **kwargs):

        one_batch_generator = self.keras_generator(training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, just_one_batch=True, batch_size=training_batch_size))

        if training_steps_per_epoch is None:
            training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1

        try:
            self.model.fit_generator(generator=one_batch_generator, steps_per_epoch=training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, callbacks=get_callbacks(callbacks=callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, model=self, batch_size=training_batch_size, backend='keras', **kwargs))
        except KeyboardInterrupt:
            pass
        except:
            raise

        one_batch = next(one_batch_generator)
        prediction = self.predict(one_batch[0])

        if show_results:
            check_data(output_data={self.input_data: one_batch[0], self.targets: one_batch[1], 'prediction': prediction}, batch_size=training_batch_size)

        return

    def create_data_generators(self, training_data_collection, validation_data_collection=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None):

        super(KerasModel, self).create_data_generators(training_data_collection, validation_data_collection, input_groups, training_batch_size, validation_batch_size, training_steps_per_epoch, validation_steps_per_epoch)

        self.training_data_generator = self.keras_generator(self.training_data_generator)

        if self.validation_data_generator is not None:
            self.validation_data_generator = self.keras_generator(self.validation_data_generator)

        return

    def keras_generator(self, data_generator, input_data='input_data', targets='ground_truth'):

        while True:
            data = next(data_generator)
            keras_data = (data[self.input_data], data[self.targets])
            yield keras_data

        return

    def predict(self, input_data):

        return self.model.predict(input_data)

    def build(self):

        self.model_input_shape = self.model.layers[0].input_shape
        self.model_output_shape = self.model.layers[-1].output_shape


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
        add_parameter(self, kwargs, 'callbacks')

        self.callbacks = get_callbacks(backend='tensorflow', model=self, batch_size=self.training_batch_size, **kwargs)

        self.init_sess()
        self.build_tensorflow_model(self.training_batch_size)
        self.create_data_generators(training_data_collection, training_batch_size=self.training_batch_size, training_steps_per_epoch=self.training_steps_per_epoch)

        return

    def train(self, training_data_collection, validation_data_collection=None, **kwargs):

        self.init_training(training_data_collection, kwargs)

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.callback_process('on_train_begin')

        for epoch in range(self.num_epochs):

            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            self.callback_process('on_epoch_begin', epoch)

            step_counter = tqdm(list(range(self.training_steps_per_epoch)), total=self.training_steps_per_epoch, unit="step", desc="Generator Loss:", miniters=1)

            for step in step_counter:

                self.callback_process('on_batch_begin', step)

                self.process_step(step_counter)

                self.callback_process('on_batch_end', step)

            self.callback_process('on_epoch_end', epoch)

        self.callback_process('on_train_end')

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

        # builder = tf.saved_model.builder.SavedModelBuilder(output_model_filepath)
        # builder.add_meta_graph_and_variables(self.sess, ['tensorflow_model'])
        # builder.save()

        return save_path

    def model_summary(self):

        for layer in tf.trainable_variables():
            print layer

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
                        print(layer.name, self.graph.get_tensor_by_name(layer.name + ':0').get_shape())
                except:
                    continue


def load_old_model(model_file, backend='keras'):

    """ Loading an old keras model file. A thing wrapper around load_model
        that uses DeepNeuro's custom cost functions.
    
        TODO: Investigate application in Tensorflow.

        Parameters
        ----------
        model_file : str
            At present, a filepath to a ".h5" keras file.
        implementation : str, optional
            Specify 'keras' or 'tensorflow' implementation.
        
        Returns
        -------
        Keras model
            A Keras model object stored in the given filepath.
        
    """

    if backend == 'keras':
        custom_objects = cost_function_dict()

        return KerasModel(model=load_model(model_file, custom_objects=custom_objects))

    if backend == 'tf':
        sess = tf.Session()    
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        return sess


# def MinimalModel(DeepNeuroModel):

#     def load(self, kwargs):

#         if 'dummy_parameter' in kwargs:
#             self.depth = kwargs.get('depth')
#         else:
#             self.depth = False

#     def build_model(self):
        
#         """ A basic implementation of the U-Net proposed in https://arxiv.org/abs/1505.04597
        
#             TODO: specify optimizer

#             Returns
#             -------
#             Keras model or tensor
#                 If input_tensor is provided, this will return a tensor. Otherwise,
#                 this will return a Keras model.
#         """

#         print self.inputs.get_shape()

#         left_outputs = []

#         for level in xrange(self.depth):

#             filter_num = int(self.max_filter / (2 ** (self.depth - level)) / self.downsize_filters_factor)

#             if level == 0:
#                 left_outputs += [Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(self.inputs)]
#                 left_outputs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(left_outputs[level])
#             else:
#                 left_outputs += [MaxPooling3D(pool_size=self.pool_size)(left_outputs[level - 1])]
#                 left_outputs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(left_outputs[level])
#                 left_outputs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(left_outputs[level])

#             if self.dropout is not None and self.dropout != 0:
#                 left_outputs[level] = Dropout(self.dropout)(left_outputs[level])

#             if self.batch_norm:
#                 left_outputs[level] = BatchNormalization()(left_outputs[level])

#         right_outputs = [left_outputs[self.depth - 1]]

#         for level in xrange(self.depth):

#             filter_num = int(self.max_filter / (2 ** (level)) / self.downsize_filters_factor)

#             if level > 0:
#                 right_outputs += [UpConvolution(pool_size=self.pool_size)(right_outputs[level - 1])]
#                 right_outputs[level] = concatenate([right_outputs[level], left_outputs[self.depth - level - 1]], axis=4)
#                 right_outputs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding)(right_outputs[level])
#                 right_outputs[level] = Conv3D(int(filter_num / 2), self.filter_shape, activation=self.activation, padding=self.padding)(right_outputs[level])
#             else:
#                 continue

#             if self.dropout is not None and self.dropout != 0:
#                 right_outputs[level] = Dropout(self.dropout)(right_outputs[level])

#             if self.batch_norm:
#                 right_outputs[level] = BatchNormalization()(right_outputs[level])

#         output_layer = Conv3D(int(self.num_outputs), (1, 1, 1))(right_outputs[-1])

#         # TODO: Brainstorm better way to specify outputs
#         if self.input_tensor is not None:
#             return output_layer

#         if self.output_type == 'regression':
#             self.model = Model(inputs=self.inputs, outputs=output_layer)
#             self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

#         if self.output_type == 'binary_label':
#             act = Activation('sigmoid')(output_layer)
#             self.model = Model(inputs=self.inputs, outputs=act)
#             self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

#         if self.output_type == 'categorical_label':
#             act = Activation('softmax')(output_layer)
#             self.model = Model(inputs=self.inputs, outputs=act)
#             self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='categorical_crossentropy',
#                           metrics=['categorical_accuracy'])

#         return self.model
