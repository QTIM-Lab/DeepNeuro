import os
import sys
import numpy as np

from collections import defaultdict

from deepneuro.utilities.util import add_parameter, replace_suffix
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti


class Preprocessor(object):

    def __init__(self, data_groups=None, save_output=True, verbose=False, output_folder=None, **kwargs):

        # File-Saving Parameters
        add_parameter(self, kwargs, 'overwrite', False)
        add_parameter(self, kwargs, 'save_output', True)
        add_parameter(self, kwargs, 'output_folder', None)
        add_parameter(self, kwargs, 'return_arrary', True)

        # Input Parameters
        add_parameter(self, kwargs, 'file_input', False)

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Conversion')
        add_parameter(self, kwargs, 'preprocessor_string', '_convert')

        # Internal Parameters
        add_parameter(self, kwargs, 'verbose', True)

        # Derived Parameters

        self.array_input = True
        self.input_data = defaultdict(list)

        self.outputs = defaultdict(list)
        self.output_data = None
        self.output_array = None
        self.output_shape = None
        self.initialization = False

        # Dreams of linked lists here.
        self.next_prepreprocessor = None
        self.previous_preprocessor = None
        self.order_index = 0

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, data_collection, return_data=True):

        """ There is a lot of repeated code in the preprocessors. Think about preprocessor structures and work on this class.
        """

        self.initialize(data_collection)  # TODO: make overwrite work with initializations

        for label, data_group in data_collection.data_groups.iteritems():

            self.generate_input_data(data_collection, data_group)

            self.generate_output_filenames(data_collection, data_group)

            self.preprocess(data_group)

            if self.save_output or not self.return_array:
                output_filenames = self.save_to_file(data_group)

            self.store_outputs(data_collection, data_group)


    def generate_input_data(self, data_collection, data_group):

        self.input_data[data_group.label] = data_group.preprocessed_case


    def preprocess(self, data_group):

        # Currently a nonsense function.

        self.output_array = data_group.preprocessed_data

        # Processing happens here.

        data_group.preprocessed_data = self.output_array


    def generate_output_filenames(self, data_collection, data_group):

        self.output_filenames = []

        for file_idx, filename in enumerate(data_group.data[data_collection.current_case]):

            if self.name == 'Conversion' and (filename.endswith('.nii') or filename.endswith('.nii.gz')):
                self.output_filenames += [filename]

            if self.output_folder is None:
                if os.path.isdir(filename):
                    self.output_filenames += [os.path.join(filename, os.path.basename(os.path.dirname(filename) + self.preprocessor_string + '.nii.gz'))]
                else:
                    self.output_filenames += [replace_suffix(filename, '', self.preprocessor_string)]
            else:
                if os.path.isdir(filename):
                    self.output_filenames += [os.path.join(self.output_folder, os.path.basename(os.path.dirname(filename) + self.preprocessor_string + '.nii.gz'))]
                else:
                    self.output_filenames += [os.path.join(self.output_folder, os.path.basename(replace_suffix(filename, '', self.preprocessor_string)))]

        return

    def save_to_file(self, data_group):

        """ No idea how this will work if the amount of output files is changed in a preprocessing step
        """

        for file_idx, output_filename in self.output_filenames:
            save_numpy_2_nifti(np.squeeze(array), affine, self.output_filename)

        return

    def store_outputs(self, data_collection, data_group):

        data_dictionary = data_collection.preprocessed_cases[data_collection.current_case][data_group.label][self.name]

        data_dictionary['output_filenames'] = self.output_filenames

        if self.output_affines is not None:
            data_dictionary['output_affine'] = self.output_affines

        return

    def delete_outputs(self, data_collection):

        data_dictionary = data_collection.preprocessed_cases[data_collection.current_case][self.name]

        for key in data_dictionary.keys():
            pass

        return

    def extra(self):

        if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
            os.remove(data_group.preprocessed_case[index])

    def initialize(self, data_collection):

        # Absolute madness here. Four nested dicts, sounds like a JSON object.

        if data_collection.preprocessed_cases[data_collection.current_case].get(self.name) is None:
            data_collection.preprocessed_cases[data_collection.current_case][self.name] = defaultdict(list)

        for label, data_group in data_collection.data_groups.iteritems():
            if data_collection.preprocessed_cases[data_collection.current_case][self.name].get(label) is None:
                data_collection.preprocessed_cases[data_collection.current_case][self.name][label] = defaultdict(list)

        if self.order_index > 0:
            self.previous_preprocessor = data_collection.preprocessors[order_index - 1]

        if self.order_index != len(data_collection.preprocessors) - 1:
            self.next_prepreprocessor = data_collection.preprocessors[order_index + 1]

        return

    def reset(self):

        self.outputs = defaultdict(list)

        return

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group


class DICOMConverter(Preprocessor):

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Conversion')
        add_parameter(self, kwargs, 'preprocessor_string', '_convert')

        return

    def generate_output_filenames(self, data_collection, data_group):

        self.output_filenames = []

        for filename in data_group.data[data_collection.current_case]:

            if self.name == 'Conversion' and (filename.endswith('.nii') or filename.endswith('.nii.gz')):
                self.output_filenames += [filename]


    def save_to_file(self, data_group):

        """ No idea how this will work if the amount of output files is changed in a preprocessing step
        """

        if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):

            output_filenames = self.generate_output_filenames(self, data_group)

            for output_filename in output_filenames:
                save_numpy_2_nifti(np.squeeze(array), affine, self.output_filename)

        return