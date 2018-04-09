
import subprocess
import os
import sys

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, replace_suffix

FNULL = open(os.devnull, 'w')


class Resample(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])
        add_parameter(self, kwargs, 'name', 'Resample')

        # Registration Parameters
        add_parameter(self, kwargs, 'dimensions', [1, 1, 1])
        add_parameter(self, kwargs, 'interpolation', 'linear')
        add_parameter(self, kwargs, 'reference_channel', None)

        # Derived Parameters
        add_parameter(self, kwargs, 'preprocessor_string', '_Resampled_' + str(self.dimensions).strip('[]').replace(' ', '').replace(',', ''))
        self.interpolation_dict = {'nearestNeighbor': 'nn', 'linear': 'linear'}
        self.dimensions = str(self.dimensions).strip('[]').replace(' ', '')

        self.array_input = False

    def preprocess(self, data_group):

        for file_idx, filename in enumerate(data_group.preprocessed_case):
            if self.reference_channel is None:
                specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, '-s', self.dimensions, filename, self.output_filenames[file_idx]]
            else:
                specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_channel, '--interpolation', self.interpolation_dict[self.interpolation], self.base_file, self.output_filename]
            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        self.output_data = self.output_filenames
        data_group.preprocessed_case = self.output_filenames

class Coregister(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'command', ['Slicer', '--launch', 'BRAINSFit'])
        add_parameter(self, kwargs, 'name', 'Registration')
        add_parameter(self, kwargs, 'preprocessor_string', '_Registered')

        # Transform Parameters
        add_parameter(self, kwargs, 'transform_type', 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine')
        add_parameter(self, kwargs, 'transform_initialization', 'useMomentsAlign')
        add_parameter(self, kwargs, 'interpolation', 'Linear')
        add_parameter(self, kwargs, 'sampling_percentage', .06)

        # Reference Parameters
        add_parameter(self, kwargs, 'reference_channel', None)
        add_parameter(self, kwargs, 'reference_file', None)

        self.interpolation_dict = {'nearestNeighbor': 'nn'}

        self.array_input = False

    def preprocess(self, data_group):

        for file_idx, filename in enumerate(data_group.preprocessed_case):

            if self.reference_channel is not None:
                if self.reference_channel == file_idx:
                    self.output_filenames[file_idx] = filename
                    continue
                else:
                    specific_command = self.command + ['--fixedVolume', data_group.preprocessed_case[self.reference_channel], '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', filename, '--outputVolume', self.output_filenames[file_idx]]
            else:
                specific_command = self.command + ['--fixedVolume', '"' + self.reference_file + '"', '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', self.base_file, '--outputVolume', self.output_filename]

            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        self.output_data = self.output_filenames
        data_group.preprocessed_case = self.output_filenames