
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


class Normalization(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'ZeroMeanNormalization')
        add_parameter(self, kwargs, 'preprocessor_string', '_ZeroNorm')

        # Mask parameters
        add_parameter(self, kwargs, 'mask', None)
        add_parameter(self, kwargs, 'mask_preprocessor', None)
        add_parameter(self, kwargs, 'mask_name', 'skullstrip_mask')
        add_parameter(self, kwargs, 'mask_zeros', False)
        add_parameter(self, kwargs, 'mask_value', None)

        # Normalization Parameters
        add_parameter(self, kwargs, 'intensity_range', [-1,1])
        add_parameter(self, kwargs, 'normalize_by_channel', True)


        self.array_input = True

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case

        if self.mask is not None:
            mask_numpy = read_image_files(self.mask)[..., 0]
        elif self.mask_preprocessor is not None:
            mask_numpy = read_image_files(self.mask_preprocessor.data_dictionary[data_group.label][self.mask_name])[..., 0]
        else:
            mask_numpy = None

        if self.normalize_by_channel:
            for channel in range(data_group.preprocessed_case.shape[-1]):
                data_group.preprocessed_case[..., channel] = self.normalize(input_data[..., channel], mask_numpy)
        else:
            data_group.preprocessed_case = self.normalize(input_data, mask_numpy)

        # TODO: Reduce redundancy in naming
        self.output_data = data_group.preprocessed_case

    def normalize(self, normalize_numpy, mask_numpy=None):

        return normalize_numpy


class RangeNormalization(Normalization):

    def load(self, kwargs):

        super(RangeNormalization, self).load(kwargs)

        add_parameter(self, kwargs, 'intensity_range', [-1,1])
        add_parameter(self, kwargs, 'input_intensity_range', None)

        add_parameter(self, kwargs, 'outlier_percent', None)  # Not Implemented

    def normalize(self, normalize_numpy, mask_numpy=None):

        normalize_numpy = normalize_numpy.astype(float)

        if mask_numpy is not None:
            mask = mask_numpy > 0
        elif self.mask_zeros:
            mask = np.nonzero(normalize_numpy)
        else:
            mask = None

        if mask is None:
            
            if self.input_intensity_range is None:
                input_intensity_range = [np.min(normalize_numpy), np.max(normalize_numpy)]
            else:
                input_intensity_range = self.input_intensity_range

            normalize_numpy = ((self.intensity_range[1] - self.intensity_range[0]) * (normalize_numpy - input_intensity_range[0])) / (input_intensity_range[1] - input_intensity_range[0]) + self.intensity_range[0] 
            
            if self.input_intensity_range is not None:
                normalize_numpy[normalize_numpy < self.intensity_range[0]] = self.intensity_range[0]
                normalize_numpy[normalize_numpy > self.intensity_range[1]] = self.intensity_range[1]
       
        else:
            raise NotImplementedError

        for channel in range(normalize_numpy.shape[-1]):
            print np.min(normalize_numpy[..., channel]), np.max(normalize_numpy[..., channel])

        return normalize_numpy


class BinaryNormalization(Normalization):

    def load(self, kwargs):

        super(BinaryNormalization, self).load(kwargs)

        add_parameter(self, kwargs, 'intensity_range', [-1,1])
        
        # Not Implemented
        add_parameter(self, kwargs, 'threshold', 0)
        add_parameter(self, kwargs, 'single_value', None)

    def normalize(self, normalize_numpy, mask_numpy=None):

        normalize_numpy = normalize_numpy.astype(float)

        if mask_numpy is not None:
            mask = mask_numpy > 0
        elif self.mask_zeros:
            mask = np.nonzero(normalize_numpy)
        else:
            mask = None

        if mask is None:
            
            if self.single_value is None:
                normalize_numpy[normalize_numpy <= self.threshold] = self.intensity_range[0]
                normalize_numpy[normalize_numpy > self.threshold] = self.intensity_range[1]
            else:
                normalize_numpy[normalize_numpy != self.single_value] = self.intensity_range[0]
                normalize_numpy[normalize_numpy == self.single_value] = self.intensity_range[1]

        else:
            raise NotImplementedError

        return normalize_numpy


class ZeroMeanNormalization(Normalization):

    def load(self, kwargs):

        super(ZeroMeanNormalization, self).load(kwargs)

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'ZeroMeanNormalization')
        add_parameter(self, kwargs, 'preprocessor_string', '_ZeroNorm')

    def normalize(self, normalize_numpy, mask_numpy=None):

        normalize_numpy = normalize_numpy.astype(float)

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

        print np.unique(normalize_numpy)
        return normalize_numpy