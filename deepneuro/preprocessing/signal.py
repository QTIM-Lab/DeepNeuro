
import subprocess
import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files
from deepneuro.utilities.util import add_parameter

FNULL = open(os.devnull, 'w')


class N4BiasCorrection(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'N4BiasCorrection')
        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])
        add_parameter(self, kwargs, 'preprocessor_string', '_N4Bias')

        self.array_input = False

    def preprocess(self, data_group):

        # specific_command = self.command + ['-i', self.base_file, '-o', self.output_filename]

        for file_idx, filename in enumerate(data_group.preprocessed_case):
            specific_command = self.command + ['N4ITKBiasFieldCorrection', filename, self.output_filenames[file_idx]]
            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        self.output_data = self.output_filenames


class ZeroMeanNormalization(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'ZeroMeanNormalization')
        add_parameter(self, kwargs, 'preprocessor_string', '_ZeroNorm')

        # Mask parameters
        add_parameter(self, kwargs, 'mask', None)
        add_parameter(self, kwargs, 'mask_preprocessor', None)
        add_parameter(self, kwargs, 'mask_name', 'skullstrip_mask')
        add_parameter(self, kwargs, 'mask_zeros', True)

        # Normalization Parameters
        add_parameter(self, kwargs, 'normalize_by_channel', True)

        self.array_input = True

    def preprocess(self, data_group):

        if self.mask is not None:
            mask_numpy = read_image_files(self.mask)[..., 0]
        elif self.mask_preprocessor is not None:
            mask_numpy = read_image_files(self.mask_preprocessor.data_dictionary[data_group.label][self.mask_name])[..., 0]
        else:
            mask_numpy = None

        if self.normalize_by_channel:
            for channel in xrange(data_group.preprocessed_case.shape[-1]):
                data_group.preprocessed_case[..., channel] = self.normalize(data_group.preprocessed_case[..., channel], mask_numpy)
        else:
            data_group.preprocessed_case = self.normalize(data_group.preprocessed_case, mask_numpy)

        # TODO: Reduce redundancy in naming
        self.output_data = data_group.preprocessed_case

    def normalize(self, normalize_numpy, mask_numpy=None):

        if mask_numpy is not None:
            vol_mean = np.mean(normalize_numpy[mask_numpy > 0])
            vol_std = np.std(normalize_numpy[mask_numpy > 0])
            normalize_numpy = (normalize_numpy - vol_mean) / vol_std
            normalize_numpy[mask_numpy == 0] = 0
        elif self.mask_zeros:
            idx_nonzeros = np.nonzero(normalize_numpy)
            vol_mean = np.mean(normalize_numpy[idx_nonzeros])
            vol_std = np.std(normalize_numpy[idx_nonzeros])
            normalize_numpy[idx_nonzeros] = (normalize_numpy[idx_nonzeros] - vol_mean) / vol_std
        else:
            vol_mean = np.mean(normalize_numpy)
            vol_std = np.std(normalize_numpy)
            normalize_numpy = (normalize_numpy - vol_mean) / vol_std

        return normalize_numpy