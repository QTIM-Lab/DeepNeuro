import os
import sys
import numpy as np

from collections import defaultdict

from deepneuro.utilities.util import add_parameter, replace_suffix
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti


class Preprocessor(object):

    def __init__(self, **kwargs):

        # File-Saving Parameters
        add_parameter(self, kwargs, 'overwrite', True)
        add_parameter(self, kwargs, 'save_output', True)
        add_parameter(self, kwargs, 'output_folder', None)
        add_parameter(self, kwargs, 'return_array', False)

        # Input Parameters
        add_parameter(self, kwargs, 'file_input', False)

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Conversion')
        add_parameter(self, kwargs, 'preprocessor_string', '_convert')

        # Internal Parameters
        add_parameter(self, kwargs, 'verbose', True)

        # Derived Parameters
        self.array_input = True

        self.outputs = defaultdict(list)
        self.output_data = None
        self.output_affines = None
        self.output_shape = None
        self.output_filenames = []
        self.initialization = False

        # Dreams of linked lists here.
        self.data_dictionary = None
        self.next_prepreprocessor = None
        self.previous_preprocessor = None
        self.order_index = 0

        # Get rid of this parameter somehow.
        self.saved = False

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, data_collection):

        """ There is a lot of repeated code in the preprocessors. Think about preprocessor structures and work on this class.
        """

        if self.verbose:
            print 'Working on Preprocessor:', self.name

        self.initialize(data_collection)  # TODO: make overwrite work with initializations

        for label, data_group in data_collection.data_groups.iteritems():

            self.generate_output_filenames(data_collection, data_group)

            self.preprocess(data_group)

            if self.save_output:
                output_filenames = self.save_to_file(data_group)

            # Duplicated code here. In general, this is pretty messy.
            if self.next_prepreprocessor is not None:
                if self.next_prepreprocessor.array_input:
                    self.convert_to_array_data(data_group)
                else:
                    self.save_to_file(data_group)
                    data_group.preprocessed_case = self.output_filenames

            if self.return_array:
                self.convert_to_array_data(data_group)

            self.store_outputs(data_collection, data_group)

    def convert_to_array_data(self, data_group):
        data_group.preprocessed_case, affine = read_image_files(self.output_data, return_affine=True)
        if affine is not None:
            data_group.preprocessed_affine = affine

    def convert_to_filename_data(self, data_group):

        return

    def preprocess(self, data_group):

        # Currently a nonsense function.

        self.output_data = data_group.preprocessed_case

        # Processing happens here.

        data_group.preprocessed_case = self.output_data

    def generate_output_filenames(self, data_collection, data_group):

        self.output_filenames = []

        for file_idx, filename in enumerate(data_group.data[data_collection.current_case]):

            self.output_filenames += [self.generate_output_filename(filename)]

        return

    def generate_output_filename(self, filename, suffix=None):

        if suffix is None:
            suffix = self.preprocessor_string

        # A bit hacky, here.
        if self.name == 'Conversion' and (filename.endswith('.nii') or filename.endswith('.nii.gz')):
            output_filename = filename
        elif self.output_folder is None:
            if os.path.isdir(filename):
                output_filename = os.path.join(filename, os.path.basename(os.path.dirname(filename) + suffix + '.nii.gz'))
            else:
                output_filename = replace_suffix(filename, '', suffix)
        else:
            if os.path.isdir(filename):
                output_filename = os.path.join(self.output_folder, os.path.basename(os.path.dirname(filename) + suffix + '.nii.gz'))
            else:
                output_filename = os.path.join(self.output_folder, os.path.basename(replace_suffix(filename, '', suffix)))

        return output_filename

    def save_to_file(self, data_group):

        """ No idea how this will work if the amount of output files is changed in a preprocessing step
            Also missing affines is a problem.
        """

        if type(self.output_data) is not list:
            for file_idx, output_filename in enumerate(self.output_filenames):
                if self.overwrite or not os.path.exists(output_filename):
                    save_numpy_2_nifti(np.squeeze(self.output_data[..., file_idx]), data_group.preprocessed_affine, output_filename)

        return

    def store_outputs(self, data_collection, data_group):

        self.data_dictionary[data_group.label]['output_filenames'] = self.output_filenames

        if self.output_affines is not None:
            self.data_dictionary[data_group.label]['output_affine'] = self.output_affines

        return

    def clear_outputs(self, data_collection, data_group, clear_files_only=False):

        for key in self.data_dictionary[data_group.label]:
            for item in self.data_dictionary[data_group.label][key]:
                if type(item) is str:
                    if os.path.exists(item):
                        if not self.save_output:
                            os.remove(item)
                elif not clear_files_only:
                    self.data_dictionary[data_group.label][key] = []
                    break

        return

    def initialize(self, data_collection):

        # Absolute madness here. Four nested dicts, sounds like a JSON object.

        self.saved = False

        if data_collection.preprocessed_cases[data_collection.current_case].get(self.name) is None:
            data_collection.preprocessed_cases[data_collection.current_case][self.name] = defaultdict(list)

        for label, data_group in data_collection.data_groups.iteritems():
            if data_collection.preprocessed_cases[data_collection.current_case][self.name].get(label) is None:
                data_collection.preprocessed_cases[data_collection.current_case][self.name][label] = defaultdict(list)

        self.data_dictionary = data_collection.preprocessed_cases[data_collection.current_case][self.name]

        if self.order_index > 0:
            self.previous_preprocessor = data_collection.preprocessors[self.order_index - 1]

        if self.order_index != len(data_collection.preprocessors) - 1:
            self.next_prepreprocessor = data_collection.preprocessors[self.order_index + 1]

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

        # Not yet implemented
        add_parameter(self, kwargs, 'output_dictionary', None)

        return