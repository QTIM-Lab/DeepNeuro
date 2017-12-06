
import subprocess
import os
import glob
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files
from deepneuro.utilities.util import add_parameter, replace_suffix

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

class N4BiasCorrection(Preprocessor):

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

        add_parameter(self, kwargs, 'command', ['N4BiasFieldCorrection'])

        self.preprocessor_string = '_N4Bias'

    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)
                specific_command = self.command + ['-i', file, '-o', output_filename]
                # subprocess.call(' '.join(specific_command), shell=True)
                
                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = output_filename

                self.outputs['outputs'] += [output_filename]

class ZeroMeanNormalization(Preprocessor):

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


        add_parameter(self, kwargs, 'mask', None)

        self.preprocessor_string = '_ZeroNorm'

    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)

                normalize_numpy = read_image_files([file])

                if self.mask is not None:
                    mask_numpy = read_image_files(self.mask.outputs['masks'])
                    vol_mean = np.mean(normalize_numpy[mask_numpy > 0])
                    vol_std = np.std(normalize_numpy[mask_numpy > 0])
                    normalize_numpy = (normalize_numpy - vol_mean) / vol_std
                    normalize_numpy[mask_numpy == 0] = 0
                else:
                    vol_mean = np.mean(normalize_numpy)
                    vol_std = np.std(normalize_numpy)
                    normalize_numpy = (normalize_numpy - vol_mean) / vol_std        

                save_numpy_2_nifti(np.squeeze(normalize_numpy), file, output_filename)

                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = output_filename
                self.outputs['outputs'] += [output_filename]