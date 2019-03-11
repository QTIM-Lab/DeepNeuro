
import subprocess
import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.conversion import read_image_files
from deepneuro.utilities.util import add_parameter, quotes

FNULL = open(os.devnull, 'w')


class N4BiasCorrection(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'N4BiasCorrection')
        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])
        add_parameter(self, kwargs, 'preprocessor_string', '_N4Bias')

        self.array_input = False

    def preprocess(self, data_group):

        for file_idx, filename in enumerate(data_group.preprocessed_case):
            specific_command = self.command + ['N4ITKBiasFieldCorrection', quotes(filename), quotes(self.output_filenames[file_idx])]
            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        self.output_data = self.output_filenames
        data_group.preprocessed_case = self.output_filenames


class MaskValues(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'MaskValues')
        add_parameter(self, kwargs, 'preprocessor_string', 'Masked')

        # Masking Parameters
        add_parameter(self, kwargs, 'mask_threshold', 0)
        add_parameter(self, kwargs, 'mask_value', 0)
        add_parameter(self, kwargs, 'mask_mode', 'lower')

        self.array_input = True

    def preprocess(self, data_group):

        if self.mask_mode == 'lower':
            data_group.preprocessed_case[data_group.preprocessed_case < self.mask_threshold] = self.mask_value
        elif self.mask_mode == 'higher':
            data_group.preprocessed_case[data_group.preprocessed_case > self.mask_threshold] = self.mask_value
        elif self.mask_mode == 'equal':
            data_group.preprocessed_case[data_group.preprocessed_case == self.mask_threshold] = self.mask_value
        else:
            raise NotImplementedError

        self.output_data = data_group.preprocessed_case


class Normalization(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Normalization')
        add_parameter(self, kwargs, 'preprocessor_string', '_Norm')

        # Mask parameters
        add_parameter(self, kwargs, 'mask', None)
        add_parameter(self, kwargs, 'mask_preprocessor', None)
        add_parameter(self, kwargs, 'mask_name', 'skullstrip_mask')
        add_parameter(self, kwargs, 'mask_zeros', False)
        add_parameter(self, kwargs, 'mask_value', None)

        # Normalization Parameters
        add_parameter(self, kwargs, 'normalize_by_channel', True)
        add_parameter(self, kwargs, 'channels', None)

        self.array_input = True

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case

        if self.mask is not None:
            mask_numpy = read_image_files(self.mask)[..., 0]
        elif self.mask_preprocessor is not None:
            mask_numpy = read_image_files(self.mask_preprocessor.mask_numpy)[..., 0]
        else:
            mask_numpy = None

        if self.channels is not None:
            return_data = np.copy(input_data).astype(float)
            input_data = np.take(input_data, indices=self.channels, axis=-1)

        if self.normalize_by_channel:
            # Make this an optional parameter.
            data_group.preprocessed_case = data_group.preprocessed_case.astype(float)
            for channel in range(data_group.preprocessed_case.shape[-1]):
                data_group.preprocessed_case[..., channel] = self.normalize(input_data[..., channel], mask_numpy)
        else:
            data_group.preprocessed_case = self.normalize(input_data, mask_numpy)

        if self.channels is not None:
            for channel_idx, channel in enumerate(self.channels):
                return_data[..., channel] = data_group.preprocessed_case[..., channel_idx]
            data_group.preprocessed_case = return_data

        # TODO: Reduce redundancy in naming
        self.output_data = data_group.preprocessed_case

    def normalize(self, normalize_numpy, mask_numpy=None):

        return normalize_numpy


class RangeNormalization(Normalization):

    def load(self, kwargs):

        super(RangeNormalization, self).load(kwargs)

        add_parameter(self, kwargs, 'intensity_range', [-1, 1])
        add_parameter(self, kwargs, 'input_intensity_range', None)

        add_parameter(self, kwargs, 'outlier_percent', None)  # Not Implemented

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'RangeNormalization')
        add_parameter(self, kwargs, 'preprocessor_string', '_Range')

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

            # Edge Case -- Minimum and Maximum
            if input_intensity_range[0] == input_intensity_range[1]:
                normalize_numpy[:] = self.intensity_range[0]
                print('Warning: normalization edge case. All array values are equal. Normalizing to minimum value.')

            else:
                normalize_numpy = ((self.intensity_range[1] - self.intensity_range[0]) * (normalize_numpy - input_intensity_range[0])) / (input_intensity_range[1] - input_intensity_range[0]) + self.intensity_range[0] 
            
            if self.input_intensity_range is not None:
                normalize_numpy[normalize_numpy < self.intensity_range[0]] = self.intensity_range[0]
                normalize_numpy[normalize_numpy > self.intensity_range[1]] = self.intensity_range[1]
       
        else:
            raise NotImplementedError

        return normalize_numpy


class BinaryNormalization(Normalization):

    def load(self, kwargs):

        super(BinaryNormalization, self).load(kwargs)

        add_parameter(self, kwargs, 'intensity_range', [-1, 1])
        add_parameter(self, kwargs, 'threshold', 0)
        add_parameter(self, kwargs, 'single_value', None)

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'BinaryNormalization')
        add_parameter(self, kwargs, 'preprocessor_string', '_Binary')

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

        return normalize_numpy