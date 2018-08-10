
import subprocess
import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti
from deepneuro.utilities.util import add_parameter, quotes

FNULL = open(os.devnull, 'w')


class SkullStrip(Preprocessor):

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

        add_parameter(self, kwargs, 'command', ['fsl4.1-bet2'])
        # add_parameter(self, kwargs, 'command', ['bet2'])

        add_parameter(self, kwargs, 'same_mask', True)
        add_parameter(self, kwargs, 'reference_channel', None)

        add_parameter(self, kwargs, 'bet2_f', .5)
        add_parameter(self, kwargs, 'bet2_g', 0)

        add_parameter(self, kwargs, 'name', 'SkullStrip')
        add_parameter(self, kwargs, 'preprocessor_string', '_SkullStripped')

        self.array_input = True

        self.mask_string = '_Skullstrip_Mask'
        self.mask_filename = None

    def initialize(self, data_collection):

        super(SkullStrip, self).initialize(data_collection)

        for label, data_group in data_collection.data_groups.items():
            
            reference_filename = data_group.data[data_collection.current_case][self.reference_channel]

            self.mask_filename = self.generate_output_filename(reference_filename, self.mask_string)

            if type(data_group.preprocessed_case) is list:
                input_file = data_group.preprocessed_case[self.reference_channel]
            else:
                # What to do about affines here... Also, reroute this file to a temporary directory.
                input_file = save_numpy_2_nifti(data_group.preprocessed_case[..., self.reference_channel], data_group.preprocessed_affine, self.generate_output_filename(reference_filename))

            specific_command = self.command + [quotes(input_file), quotes(self.mask_filename), '-f', str(self.bet2_f), '-g', str(self.bet2_g), '-m']

            subprocess.call(' '.join(specific_command), shell=True)
            os.rename(self.mask_filename + '_mask.nii.gz', self.mask_filename)

        self.mask_numpy = read_image_files(self.mask_filename, return_affine=False)

    def preprocess(self, data_group):

        self.output_data = data_group.preprocessed_case

        # Ineffective numpy broadcasting happening here..
        self.output_data[self.mask_numpy[..., 0] == 0] = 0

        data_group.preprocessed_data = self.output_data

    def store_outputs(self, data_collection, data_group):

        self.data_dictionary[data_group.label]['skullstrip_mask'] = [self.mask_filename]

        return   


class SkullStrip_Model(Preprocessor):

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

        add_parameter(self, kwargs, 'same_mask', True)
        add_parameter(self, kwargs, 'reference_channel', [0, 1])
        add_parameter(self, kwargs, 'model', None)  # TODO: Replace with load(skull_strip_model)

        add_parameter(self, kwargs, 'name', 'SkullStrip_Model')
        add_parameter(self, kwargs, 'preprocessor_string', '_SkullStripped')

        self.array_input = True

        self.mask_string = '_Skullstrip_Mask'
        self.mask_filename = None

        if type(self.reference_channel) is not list:
            self.reference_channel = [self.reference_channel]

    def initialize(self, data_collection):

        super(SkullStrip_Model, self).initialize(data_collection)

        for label, data_group in data_collection.data_groups.items():

            reference_filename = data_group.data[data_collection.current_case][self.reference_channel[0]]
            self.mask_filename = self.generate_output_filename(reference_filename, self.mask_string)

            input_data = np.take(data_group.preprocessed_case, self.reference_channel, axis=-1)[np.newaxis, ...]

            # Also Hacky
            self.model.outputs[-1].model = self.model
            self.model.outputs[-1].input_patch_shape = self.model.outputs[-1].model.model.layers[0].input_shape
            self.model.outputs[-1].process_case([input_data])
            self.model.outputs[-1].postprocess()
            save_numpy_2_nifti(np.squeeze(self.model.outputs[-1].return_objects[-1]), data_group.preprocessed_affine, self.mask_filename)  # Hacky

        self.mask_numpy = read_image_files(self.mask_filename, return_affine=False)

    def preprocess(self, data_group):

        self.output_data = data_group.preprocessed_case

        # Ineffective numpy broadcasting happening here..
        self.output_data[self.mask_numpy[..., 0] == 0] = 0

        data_group.preprocessed_data = self.output_data

    def store_outputs(self, data_collection, data_group):

        self.data_dictionary[data_group.label]['skullstrip_mask'] = [self.mask_filename]

        return   