""" model.py specified the basic DeepNeuroModel class, as well as some model utilites and
    pre-built model chunks that may be useful across all models.
"""

from keras.engine import Input
from keras.models import load_model
from keras.layers import UpSampling3D
from keras.callbacks import ModelCheckpoint

from deepneuro.models.cost_functions import cost_function_dict


class DeepNeuroModel(object):
    
    def __init__(self, model=None, input_shape=(32, 32, 32, 1), input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), filter_shape=(3, 3, 3), dropout=.1, batch_norm=False, initial_learning_rate=0.00001, output_type='regression', num_outputs=1, activation='relu', padding='same', implementation='keras', **kwargs):

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

        self.input_shape = input_shape
        self.input_tensor = input_tensor

        if input_tensor is None:
            self.inputs = Input(input_shape)
        else:
            self.inputs = input_tensor

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

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model'], **kwargs):

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

        self.model.save(output_model_filepath)

    def append_output(self, outputs):

        for output in outputs:
            self.outputs += [output]

    def generate_outputs(self):

        for output in self.outputs:
            output.model = self
            output.execute()

def get_callbacks(model_file, callbacks=['save_model'], monitor='loss', kwargs={}):

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
            return_callbacks += [ ModelCheckpoint(model_file, monitor=monitor, save_best_only=save_best_only)]

    # filepath="weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"

    return return_callbacks


def UpConvolution(deconvolution=False, pool_size=(2, 2, 2), implementation='keras'):

    """ Keras doesn't have native support for deconvolution yet, but keras_contrib does.
        If deconvolution is not specified, normal upsampling will be used.

        TODO: Currently only works in 2D.
    
        Parameters
        ----------
        deconvolution : bool, optional
            If true, will attempt to load Deconvolutio from keras_contrib
        pool_size : tuple, optional
            Upsampling ratio along each axis.
        implementation : str, optional
            Specify 'keras' or 'tensorflow' implementation.
        
        Returns
        -------
        Keras Tensor Operation
            Either Upsampling3D() or Deconvolution()
    """

    if implementation == 'keras':
        if not deconvolution:
            return UpSampling3D(size=pool_size)
        else:
            return None
            # deconvolution not yet implemented // required from keras_contrib


def load_old_model(model_file, implementation='keras'):

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

    custom_objects = cost_function_dict()

    return DeepNeuroModel(model = load_model(model_file, custom_objects=custom_objects))
