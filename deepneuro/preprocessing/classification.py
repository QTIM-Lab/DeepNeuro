import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter

FNULL = open(os.devnull, 'w')


class MapClasses(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'MapClasses')

        # Class Parameters
        add_parameter(self, kwargs, 'class_dictionary', {1: 1, 2: 1, 3: 2, 4: 2})

        # Dervied Parameters
        self.output_classes = np.unique(list(self.class_dictionary.values()))

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(MapClasses, self).initialize(data_collection)

        for label, data_group in list(self.data_groups.items()):
            data_shape = list(data_group.get_shape())
            data_shape[-1] = len(self.output_classes)
            self.output_shape[label] = tuple(data_shape)

    def preprocess(self, data_group):

        # Relatively brittle, only works for 1-dimensional data.
        # TODO: Make work for 
        input_data = data_group.preprocessed_case

        output_data = self.class_dictionary(input_data)

        data_group.preprocessed_case = output_data
        self.output_data = output_data


class OneHotEncode(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'OneHotEncode')

        # Class Parameters
        add_parameter(self, kwargs, 'num_classes', 3)
        add_parameter(self, kwargs, 'input_classes', None)
        add_parameter(self, kwargs, 'class_dictionary', {})

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(OneHotEncode, self).initialize(data_collection)

        if self.class_dictionary == {} and self.input_classes is not None:
            for idx, class_name in enumerate(self.input_classes):
                self.class_dictionary[class_name] = idx

        for label, data_group in list(self.data_groups.items()):
            data_shape = list(data_group.get_shape())
            data_shape[-1] = self.num_classes
            self.output_shape[label] = tuple(data_shape)

    def preprocess(self, data_group):

        # Relatively brittle, only works for 1-dimensional data.
        input_data = data_group.preprocessed_case

        # Probably not the most efficient.
        output_data = np.zeros(self.num_classes)
        for item in input_data:
            if self.class_dictionary != {}:
                output_data[self.class_dictionary[item]] = 1
            else:
                output_data[int(item)] = 1

        data_group.preprocessed_case = output_data
        self.output_data = output_data


class OrdinalClasses(OneHotEncode):

    def preprocess(self, data_group):

        # Relatively brittle, only works for 1-dimensional data.
        input_data = data_group.preprocessed_case

        # Probably not the most efficient.
        output_data = np.zeros(self.num_classes)
        for item in input_data:
            if self.class_dictionary != {}:
                output_data[0:self.class_dictionary[item] + 1] = 1
            else:
                output_data[0:int(item) + 1] = 1

        data_group.preprocessed_case = output_data
        self.output_data = output_data


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
