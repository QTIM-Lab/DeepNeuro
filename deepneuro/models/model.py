""" model.py specified the basic DeepNeuroModel class, as well as some model utilites and
    pre-built model chunks that may be useful across all models.
"""

import csv

from deepneuro.models.cost_functions import cost_function_dict
from deepneuro.utilities.util import add_parameter, nifti_splitext
from deepneuro.load.load import load

class DeepNeuroModel(object):
    
    def __init__(self, **kwargs):

        """A model object with some basic parameters that can be added to in the load() method. Each child of
        this class should be able to build and store a model composed of tensors, as well as convert an input
        tensor into an output tensor according to a model's schema. 
        
        Parameters
        ----------
        input_shape : tuple, optional
            Input dimensions of first layer. Not counting batch-size.
        input_tensor : tensor, optional
            If input_tensor is specified, will output a tensor
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
        output_type, cost_function : str, optional
            Currently can choose from 'regression', which is an output the same size as input,
            'binary_label' which is a binary probability map, or 'categorical_label'
        num_outputs : int, optional
            If output_type is 'categorical_label', this specified how many outputs.
        activation : str, optional
            What type of activation to use at each layer. May be implemented differently in
            each model.
        padding : str, optional
            Padding for convolutional layers.
        backend : str, optional
            Determines deep learning framework for model implementation, if multiple available.
        **kwargs
            Addtional variables that may be needed in children classes.
        """

        # Inputs
        add_parameter(self, kwargs, 'model', None)
        add_parameter(self, kwargs, 'input_shape', (32, 32, 32, 1))
        add_parameter(self, kwargs, 'input_tensor', None)
        add_parameter(self, kwargs, 'dim', len(self.input_shape) - 1)
        add_parameter(self, kwargs, 'channels', self.input_shape[-1])

        # Generic Model Parameters -- Optional
        add_parameter(self, kwargs, 'pool_size', (2, 2, 2))
        add_parameter(self, kwargs, 'filter_shape', (3, 3, 3))
        add_parameter(self, kwargs, 'padding', 'same')
        add_parameter(self, kwargs, 'downsize_filters_factor', 1)
        add_parameter(self, kwargs, 'max_filter', 512)
        add_parameter(self, kwargs, 'kernel_size', (3, 3, 3))
        add_parameter(self, kwargs, 'stride_size', (1, 1, 1))
        add_parameter(self, kwargs, 'activation', 'relu')
        add_parameter(self, kwargs, 'optimizer', 'Nadam')
        add_parameter(self, kwargs, 'output_type', None)
        add_parameter(self, kwargs, 'cost_function', 'mse')
        add_parameter(self, kwargs, 'dropout', .1)
        add_parameter(self, kwargs, 'batch_norm', True)
        add_parameter(self, kwargs, 'initial_learning_rate', .0001)
        add_parameter(self, kwargs, 'num_outputs', 1)

        # Logging Parameters - Temporary
        add_parameter(self, kwargs, 'output_log_file', 'deepneuro_log.csv')
        add_parameter(self, kwargs, 'tensorboard_directory', None)
        add_parameter(self, kwargs, 'tensorboard_run_directory', None)

        # DeepNeuro Parameters
        add_parameter(self, kwargs, 'input_data', 'input_data')
        add_parameter(self, kwargs, 'targets', 'ground_truth')

        # Misc
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'hyperverbose', False)
        add_parameter(self, kwargs, 'initial_build', True)

        # Callbacks -- Refactor later.
        self.callbacks = []

        # Derived Parameters
        self.write_file = None
        self.csv_writer = None

        # TODO: Phase out 'output_type' in favor of 'cost_function'
        if self.cost_function is not None:
            self.output_type = self.cost_function

        self.load(kwargs)

        if self.model is None and self.initial_build:
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

    def generate_outputs(self, data_collection=None, case=None):

        return_outputs = []

        for output in self.outputs:
            # A little odd.
            output.model, output.data_collection, output.case = self, data_collection, case
            return_outputs += [output.generate()]

        return return_outputs

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

        return input_data

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
                    print(('Logging Output', input_idx, single_input))
                else:
                    print((headers[input_idx], single_input))

        return

    def close_model(self):

        if self.write_file is not None:
            if self.write_file.open:
                self.write_file.close()


def load_old_model(model_file, backend='keras', model_name=None, custom_object_dict=None, **kwargs):

    """ Loading an old keras model file. A thing wrapper around load_model
        that uses DeepNeuro's custom cost functions.

        Parameters
        ----------
        model_file : str
            At present, a filepath to a ".h5" keras file.
        backend : str, optional
            Specify 'keras' or 'tensorflow' backend for loading.
        model_name : str, optional
            If loading a Tensorflow model, you must specify which
            DeepNeuroModel architecture is loaded.
        custom_object_dict: dict, optional
            If you are loading a model with custom objects.
        
        Returns
        -------
        Keras model
            A Keras model object stored in the given filepath.
        
    """

    if backend == 'keras':

        from keras.models import load_model, model_from_json
        from deepneuro.models.keras_model import KerasModel

        custom_objects = cost_function_dict(**kwargs)

        model_file_extension = nifti_splitext(model_file)[1]

        if model_file_extension == '.json':
            model = KerasModel(initial_build=False, **kwargs)
            with open(model_file, 'r') as json_file:
                loaded_model_json = json_file.read()
                model.model = model_from_json(loaded_model_json, custom_objects=custom_objects)
            model.build_model(compute_output=False)
        else:
            model = KerasModel(model=load_model(model_file, custom_objects=custom_objects))
        
        # Necessary?
        model.build_model(compute_output=False)

        return model

    if backend == 'tf' or backend == 'tensorflow':

        import tensorflow as tf

        sess = tf.Session()    
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(model_file)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        return sess


def load_model_with_output(model_path=None, model_name=None, outputs=None, postprocessors=None, **kwargs):

    if model_path is not None:
        model = load_old_model(model_path, **kwargs)

    elif model_name is not None:
        model = load_old_model(load(model_name), **kwargs)

    else:
        print('Error. No model provided.')
        return
    
    for output in outputs:
        model.append_output([output])

        for postprocessor in postprocessors:
            output.append_postprocessor([postprocessor]) 

    return model