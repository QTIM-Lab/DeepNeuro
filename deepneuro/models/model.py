""" model.py specified the basic DeepNeuroModel class, as well as some model utilites and
    pre-built model chunks that may be useful across all models.
"""

from keras.engine import Input
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from deepneuro.models.cost_functions import cost_function_dict
from deepneuro.utilities.util import add_parameter

import tensorflow as tf


class DeepNeuroModel(object):
    
    def __init__(self, model=None, downsize_filters_factor=1, pool_size=(2, 2, 2), filter_shape=(3, 3, 3), dropout=.1, batch_norm=False, initial_learning_rate=0.00001, output_type='regression', num_outputs=1, activation='relu', padding='same', implementation='keras', **kwargs):

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

        if self.input_tensor is None:
            self.inputs = Input(self.input_shape)
        else:
            self.inputs = input_tensor

        # Generic Model Parameters -- Optional
        self.pool_size = pool_size
        self.filter_shape = filter_shape
        self.padding = padding
        self.downsize_filters_factor = downsize_filters_factor

        self.dropout = dropout
        self.batch_norm = batch_norm

        self.activation = activation

        self.initial_learning_rate = initial_learning_rate

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

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model','early_stopping','log'], **kwargs):

        """
        input_groups : list of strings, optional
            Specifies which named data groups (e.g. "ground_truth") enter which input
            data slot in your model.
        """

        # Todo: investigate call-backs more thoroughly.
        # Also, maybe something more general for the difference between training and validation.
        # Todo: list-checking for callbacks

        if training_steps_per_epoch is None:
            training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1

        training_data_generator = training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=training_batch_size)

        if validation_data_collection is None:

            self.model.fit_generator(generator=training_data_generator, steps_per_epoch=training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, callbacks=get_callbacks(output_model_filepath, callbacks=callbacks, kwargs=kwargs))

        else:

            if validation_steps_per_epoch is None:
                validation_steps_per_epoch = validation_data_collection.total_cases // validation_batch_size + 1

            validation_data_generator = validation_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=validation_batch_size)

            self.model.fit_generator(generator=training_data_generator, steps_per_epoch=training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, validation_data=validation_data_generator, validation_steps=validation_steps_per_epoch, callbacks=get_callbacks(output_model_filepath, callbacks=callbacks, kwargs=kwargs))

        # self.model.save(output_model_filepath)

    def append_output(self, outputs):

        for output in outputs:
            self.outputs += [output]

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

        self.training_data_generator = training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=training_batch_size)

        if validation_data_collection is not None:

            if validation_steps_per_epoch is None:
                self.validation_steps_per_epoch = validation_data_collection.total_cases // validation_batch_size + 1

            self.validation_data_generator = validation_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, batch_size=validation_batch_size)

    def predict(self, input_data):

        self.model.predict(input_data)

class KerasModel(DeepNeuroModel):

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model','early_stopping','log'], **kwargs):

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

            self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, callbacks=get_callbacks(output_model_filepath, callbacks=callbacks, kwargs=kwargs))

        else:

            self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, validation_data=self.validation_data_generator, validation_steps=self.validation_steps_per_epoch, callbacks=get_callbacks(output_model_filepath, callbacks=callbacks, kwargs=kwargs))


tensorflow_optimizer_dict = {'Adam', tf.train.AdamOptimizer}


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

        # Basic Model Parameters
        add_parameter(self, kwargs, 'optimizer', 'Adam')
        add_parameter(self, kwargs, 'learning_rate', 'Adam')

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model'], **kwargs):

        self.create_data_generators(training_data_collection, validation_data_collection, input_groups, training_batch_size, validation_batch_size, training_steps_per_epoch, validation_steps_per_epoch)

        from pprint import pprint
        one_item = next(self.training_data_generator)
        pprint(len(one_item))
        pprint(one_item[1].shape)

    def get_optimizer(self):

        return

    def init_sess(self):

        if self.sess is None:
            self.init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=config)

            self.sess.run(self.init)

        elif self.sess._closed:
            self.sess.run(self.init)

    def save_model(self):

        return


def get_callbacks(model_file, callbacks=['save_model','early_stopping','log'], monitor='val_loss', kwargs={}):

    """ Temporary function; callbacks will be dealt with in more detail in the future.
        Very disorganized currently. Do with dictionary. 
    """

    if 'save_best_only' in kwargs:
        save_best_only = kwargs.get('save_best_only')
    else:
        save_best_only = True

    return_callbacks = []
    for callback in callbacks:
        if callback == 'save_model':
            return_callbacks += [ModelCheckpoint(model_file, monitor=monitor, save_best_only=save_best_only)]
        if callback == 'early_stopping':
            return_callbacks += [EarlyStopping(monitor=monitor, patience=10)]
        if callback == 'log':
            return_callbacks += [CSVLogger(model_file.replace('.h5','.log'))]
    return return_callbacks


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

        loaded_model = KerasModel(model=load_model(model_file, custom_objects=custom_objects))

        self.input_shape = loaded_model.layers[0].input_shape
        self.output_shape = loaded_model.layers[-1].output_shape

        return loaded_model

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
