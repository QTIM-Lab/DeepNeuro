
from keras.engine import Input
from keras.models import load_model

from deepneuro.models.cost_functions import cost_function_dict

class DeepNeuroModel(object):

    def __init__(self, downsize_filters_factor=1, pool_size=(2, 2, 2), filter_shape=(3,3,3), dropout=.1, batch_norm=False, initial_learning_rate=0.00001, activation='relu', padding='same', **kwargs):

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

        self.activation = self.activation

        self.initial_learning_rate = self.initial_learning_rate

        self.model = None

        self.load(**kwargs)

        pass

    def load(self):

        pass

    def build_model(self):

        pass

    def model(self):

        if model is None:
            self.build_model()

        return self.model


def UpConvolution(deconvolution=False, pool_size=(2,2,2)):

    if not deconvolution:
        return UpSampling3D(size=pool_size, data_format='channels_first')

    # deconvolution not yet implemented // required from keras_contrib


def load_old_model(model_file):

    custom_objects = cost_function_dict

    return load_model(model_file, custom_objects=custom_objects)
