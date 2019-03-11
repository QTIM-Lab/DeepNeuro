import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter

FNULL = open(os.devnull, 'w')


class GetCentroid(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'OneHotClasses')

        # Dropping Parameters
        add_parameter(self, kwargs, 'channels', None)
        add_parameter(self, kwargs, 'output_data_group', None)
        add_parameter(self, kwargs, 'max_centroid', 1)
        add_parameter(self, kwargs, 'aggregate_centroid', False)

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(GetCentroid, self).initialize(data_collection)

        for label, data_group in list(self.data_groups.items()):

            data_shape = list(data_group.get_shape())

            if self.channels is None:
                self.output_shape[label] = (len(data_shape) - 1, self.max_centroid, 1)
            else:
                if type(self.channels) is not list:
                    self.channels = [self.channels]
                self.output_shape[label] = (len(data_shape) - 1, self.max_centroid, len(self.channels))

    def preprocess(self, data_group):

        raise NotImplementedError

        if self.channels is None:
            pass

        input_data = data_group.preprocessed_case
        output_data = np.take(input_data, self.channels, axis=-1)

        data_group.preprocessed_case = output_data
        self.output_data = output_data
