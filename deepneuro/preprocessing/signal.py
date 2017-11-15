
import subprocess
import os
import glob

from deepneuro.preprocessing.preprocessor import Preprocessor

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti
from qtim_tools.qtim_utilities.file_util import replace_suffix, nifti_splitext

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

        if 'command' in kwargs:
            self.command = kwargs.get('command')
        else:
            self.command = ['N4BiasFieldCorrection']

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