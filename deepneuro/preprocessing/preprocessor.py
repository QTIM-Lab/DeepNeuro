import os
import numpy as np

from deepneuro.utilities.util import add_parameter, replace_suffix, cli_sanitize, docker_print
from deepneuro.utilities.conversion import read_image_files, save_data


class Preprocessor(object):

    def __init__(self, **kwargs):

        # File-Saving Parameters
        add_parameter(self, kwargs, 'overwrite', True)
        add_parameter(self, kwargs, 'save_output', False)
        add_parameter(self, kwargs, 'output_folder', None)
        
        # Input Parameters
        add_parameter(self, kwargs, 'file_input', False)

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Conversion')
        add_parameter(self, kwargs, 'preprocessor_string', '_convert')

        # Internal Parameters
        add_parameter(self, kwargs, 'data_groups', None)
        add_parameter(self, kwargs, 'verbose', False)

        # Derived Parameters
        self.array_input = True

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

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    # @profile
    def execute(self, data_collection, return_array=False):

        """ This function serves operates the main logic of preprocessors.
            Its purpose is to check if data needs to be saved to disk before
            preprocessing (e.g. by an external program), or load data from
            disk if it is to be processed in Python. After processing data
            it can similarly return data in memory or on disk.
        """

        if self.verbose:
            docker_print('Working on Preprocessor:', self.name)

        for label, data_group in self.data_groups_iterator:

            self.generate_output_filenames(data_collection, data_group)

            if self.array_input and type(data_group.preprocessed_case) is list:
                data_group.preprocessed_case, data_group.preprocessed_affine = data_group.get_data(return_affine=True)
            elif not self.array_input and type(data_group.preprocessed_case) is list:
                pass
            elif not self.array_input:
                self.output_data = data_group.preprocessed_case
                self.save_to_file(data_group)
                data_group.preprocessed_case = self.output_filenames

            self.preprocess(data_group)

            if self.save_output:
                self.save_to_file(data_group)

            if return_array:
                self.convert_to_array_data(data_group)

    def convert_to_array_data(self, data_group):

        data_group.preprocessed_case, affine = read_image_files(self.output_data, return_affine=True)

        if affine is not None:
            data_group.preprocessed_affine = affine

    def convert_to_filename_data(self, data_group):

        return

    def preprocess(self, data_group):

        # Toy function for base class.

        self.output_data = data_group.preprocessed_case
        data_group.preprocessed_case = self.output_data

    # @profile
    def generate_output_filenames(self, data_collection, data_group, file_extension='.nii.gz'):

        """ Generates logical filenames by appending input data filenames
            with preprocessor_string. These strings will accumulate over
            multiple preprocessors
        """

        self.output_filenames = []

        if data_group.source == 'hdf5':
            for channel_num in range(data_group.data[data_collection.current_case].shape[-1]):

                self.output_filenames += [self.generate_output_filename('_'.join([data_group.label, str(data_collection.current_case), str(channel_num)]) + file_extension, file_extension=file_extension)]

        else:
            for file_idx, filename in enumerate(data_group.data[data_collection.current_case]):

                self.output_filenames += [self.generate_output_filename(filename, file_extension=file_extension)]

        return

    # @profile
    def generate_output_filename(self, filename, suffix=None, file_extension='.nii.gz'):

        if suffix is None:
            suffix = self.preprocessor_string

        filename = os.path.abspath(filename)

        # A bit hacky
        if self.name == 'Conversion' and (filename.endswith('.nii') or filename.endswith('.nii.gz')):
            output_filename = filename
        elif self.output_folder is None:
            if os.path.isdir(filename):
                output_filename = os.path.join(filename, os.path.basename(os.path.dirname(filename) + suffix + file_extension))
            else:
                output_filename = replace_suffix(filename, '', suffix)
        else:
            if os.path.isdir(filename):
                output_filename = os.path.join(self.output_folder, os.path.basename(filename + suffix + file_extension))
            else:
                output_filename = os.path.join(self.output_folder, os.path.basename(replace_suffix(filename, '', suffix, file_extension=file_extension)))

        return cli_sanitize(output_filename)

    def save_to_file(self, data_group):

        if type(self.output_data) is not list:
            for file_idx, output_filename in enumerate(self.output_filenames):
                if self.overwrite or not os.path.exists(output_filename):
                    save_data(np.squeeze(self.output_data[..., file_idx]), output_filename, reference_data=data_group.preprocessed_affine)

        return

    def store_outputs(self, data_collection, data_group):

        raise NotImplementedError

        return

    def clear_outputs(self, data_collection):

        """ Remove intermediate preprocessor outputs if save_to_file is set to False.
        """

        for label, data_group in list(self.data_groups.items()):

            self.generate_output_filenames(data_collection, data_group)

            for file_idx, output_filename in enumerate(self.output_filenames):
                if os.path.exists(output_filename):
                    os.remove(output_filename)

        return

    def initialize(self, data_collection):

        """ Called by DataCollection objects when Preprocessors
            are appended.
        """

        if self.data_groups is None:
            self.data_groups = data_collection.data_groups
        else:
            self.data_groups = {label: data_group for label, data_group in list(data_collection.data_groups.items()) if label in self.data_groups}

        self.data_groups_iterator = list(self.data_groups.items())

        return

    def reset(self):

        return


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

        self.array_input = False

        return

    def execute(self, data_collection, return_array=False):

        """ There is a lot of repeated code in the preprocessors. Think about preprocessor structures and work on this class.
        """

        if self.verbose:
            docker_print('Working on Preprocessor:', self.name)

        for label, data_group in list(self.data_groups.items()):

            self.generate_output_filenames(data_collection, data_group)

            if type(data_group.preprocessed_case) is not list:
                self.output_data = data_group.preprocessed_case            
            else:

                for file_idx, output_filename in enumerate(self.output_filenames):
                    if os.path.isdir(data_group.preprocessed_case[file_idx]):
                        if self.overwrite or not os.path.exists(output_filename):
                            array_data, affine = read_image_files(data_group.preprocessed_case[file_idx], return_affine=True)
                            # TO-DO: Check if subsetting language behaviour below has edge cases.
                            save_data(array_data[..., 0], output_filename, reference_data=affine)
                    else:
                        self.output_filenames[file_idx] = data_group.preprocessed_case[file_idx]

                data_group.preprocessed_case = self.output_filenames
                self.output_data = data_group.preprocessed_case

            if return_array:
                self.convert_to_array_data(data_group)

