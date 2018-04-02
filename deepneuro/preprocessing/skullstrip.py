
import subprocess
import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti
from deepneuro.utilities.util import add_parameter, replace_suffix

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

        self.mask_string = '_Skullstrip_Mask'

    def initialize(self, data_collection):

        super(Preprocessor, self).initialize(self)

        for label, data_group in data_collection.data_groups.iteritems():

            if type(data_group.preprocessed_case) is list:
                input_file = data_group.preprocessed_case[self.reference_channel]
            else:
                save_numpy_2_nifti()



            output_filename = replace_suffix(input_file, '', self.mask_string)

            if self.output_folder is None:
                output_filename = replace_suffix(input_file, '', self.mask_string)
            else:
                output_filename = os.path.join(self.output_folder, os.path.basename(replace_suffix(input_file, '', self.mask_string)))

            specific_command = self.command + [input_file, output_filename, '-f', str(self.bet2_f), '-g', str(self.bet2_g), '-m']
            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
            os.rename(output_filename + '_mask.nii.gz', output_filename)

            # Outputs masks is causing a lot of problems, and doesn't fit with the data groups syntax.
            self.outputs['masks'] += [output_filename]

            data_dictionary = data_collection.preprocessed_cases[data_collection.current_case][data_group.label][self.name]
            data_dictionary['output_filenames'] = self.output_filenames

        self.mask_numpy = read_image_files(self.outputs['masks'])

    def preprocess(self):

        input_numpy = read_image_files([self.base_file])
        input_numpy[self.mask_numpy == 0] = 0

        save_numpy_2_nifti(np.squeeze(input_numpy), self.base_file, self.output_filename)

    def preprocess(self, data_group):

        if type(data_group.preprocessed_case) is list:
            input_numpy = read_image_files([self.base_file])
            self.output_array, self.output_affines = read_image_files(data_group.preprocessed_case, return_affine=True)
        else:
            self.output_array = data_group.preprocessed_data

        data_group.preprocessed_case = self.output_array

    def store_outputs(self, data_collection, data_group):

        data_dictionary = data_collection.preprocessed_cases[data_collection.current_case][data_group.label][self.name]

        data_dictionary['output_filenames'] = self.output_filenames

        if self.output_affines is not None:
            data_dictionary['output_affine'] = self.output_affines

        return   