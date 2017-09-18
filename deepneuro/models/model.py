""" model.py specified the basic DeepNeuroModel class, as well as some model utilites and
    pre-built model chunks that may be useful across all models.
"""

from keras.engine import Input
from keras.models import load_model
from keras.layers import UpSampling3D

from deepneuro.models.cost_functions import cost_function_dict


class DeepNeuroModel(object):
    
    def __init__(self, input_shape=(1, 32, 32, 32), input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), filter_shape=(3, 3, 3), dropout=.1, batch_norm=False, initial_learning_rate=0.00001, output_type='regression', num_outputs=1, activation='relu', padding='same', implementation='keras', **kwargs):

        """ A model object with some basic parameters that can be added to in the load() method. Each child of
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

        self.dropout = dropout
        self.batch_norm = batch_norm

        self.activation = activation

        self.initial_learning_rate = initial_learning_rate

        self.num_outputs = num_outputs
        self.output_type = output_type

        self.implementation = implementation

        self.model = None

        self.load(kwargs)

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

        """ Access this classes model attribute. TODO: investigate overloading.
        """

        if self.model is None:
            self.build_model()

        return self.model


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

    custom_objects = cost_function_dict

    return load_model(model_file, custom_objects=custom_objects)
