
import subprocess
import os
import glob
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files
from deepneuro.utilities.util import add_parameter, replace_suffix

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

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
        add_parameter(self, kwargs, 'same_mask', True)
        add_parameter(self, kwargs, 'reference_channel', None)

        add_parameter(self, kwargs, 'bet2_f', .5)
        add_parameter(self, kwargs, 'bet2_g', 0)

        self.preprocessor_string = '_SkullStripped'
        self.mask_string = '_Skullstrip_Mask'

    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            index, file = self.reference_channel, data_group.preprocessed_case[self.reference_channel]
            output_filename = replace_suffix(file, '', self.mask_string)
            specific_command = self.command + [file, output_filename, '-f', str(self.bet2_f), '-g', str(self.bet2_g), '-m']

            subprocess.call(' '.join(specific_command), shell=True)

            output_mask_filename = replace_suffix(file, '', self.mask_string)
            os.rename(output_filename + '_mask.nii.gz', output_mask_filename)

            self.outputs['masks'] += [output_mask_filename]

        mask_numpy = read_image_files(self.outputs['masks'])

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)

                input_numpy = read_image_files([file])
                input_numpy[mask_numpy == 0] = 0

                save_numpy_2_nifti(np.squeeze(input_numpy), file, output_filename)

                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = output_filename
                self.outputs['outputs'] += [output_filename]