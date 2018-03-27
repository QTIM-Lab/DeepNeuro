
import subprocess
import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti
from deepneuro.utilities.util import add_parameter

FNULL = open(os.devnull, 'w')


class N4BiasCorrection(Preprocessor):

    def load(self, kwargs):

        add_parameter(self, kwargs, 'name', 'N4BiasCorrection')
        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])
        add_parameter(self, kwargs, 'preprocessor_string', '_N4Bias')

    def preprocess(self):

        specific_command = self.command + ['N4ITKBiasFieldCorrection', self.base_file, self.output_filename]
        # specific_command = self.command + ['-i', self.base_file, '-o', self.output_filename]

        subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)


class ZeroMeanNormalization(Preprocessor):

    def load(self, kwargs):

        add_parameter(self, kwargs, 'name', 'ZeroMeanNormalization')
        add_parameter(self, kwargs, 'mask', None)
        add_parameter(self, kwargs, 'preprocessor_string', '_ZeroNorm')
        add_parameter(self, kwargs, 'mask_zeros', True)

    def preprocess(self):

        normalize_numpy = read_image_files([self.base_file])

        if self.mask is not None:
            mask_numpy = read_image_files(self.mask.outputs['masks'])
            vol_mean = np.mean(normalize_numpy[mask_numpy > 0])
            vol_std = np.std(normalize_numpy[mask_numpy > 0])
            normalize_numpy = (normalize_numpy - vol_mean) / vol_std
            normalize_numpy[mask_numpy == 0] = 0
        else:
            # Nonzero indices is a bit messy, maybe make a parameter?
            if self.mask_zeros:
                idx_nonzeros = np.nonzero(normalize_numpy)
                vol_mean = np.mean(normalize_numpy[idx_nonzeros])
                vol_std = np.mean(normalize_numpy[idx_nonzeros])
                normalize_numpy[idx_nonzeros] = (normalize_numpy[idx_nonzeros] - vol_mean) / vol_std
            else:
                vol_mean = np.mean(normalize_numpy)
                vol_std = np.std(normalize_numpy)
                normalize_numpy = (normalize_numpy - vol_mean) / vol_std        

        save_numpy_2_nifti(np.squeeze(normalize_numpy), self.base_file, self.output_filename)