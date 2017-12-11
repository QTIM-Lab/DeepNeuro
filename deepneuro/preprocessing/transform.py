
import subprocess
import os
import glob

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, replace_suffix
from deepneuro.utilities.conversion import save_numpy_2_nifti

FNULL = open(os.devnull, 'w')

class Resample(Preprocessor):

    def load(self, kwargs):

        add_parameter(self, kwargs, 'command', ['Slicer', '--launch'])

        add_parameter(self, kwargs, 'dimensions', [1,1,1])
        add_parameter(self, kwargs, 'interpolation', 'linear')
        add_parameter(self, kwargs, 'reference_file', None)

        add_parameter(self, kwargs, 'preprocessor_string', '_Resampled_' + str(self.dimensions).strip('[]').replace(' ', '').replace(',', ''))

        self.interpolation_dict = {'nearestNeighbor': 'nn', 'linear': 'linear'}
        self.dimensions = str(self.dimensions).strip('[]').replace(' ', '')

    def preprocess():

        if self.reference_file is None:
            specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, '-s', self.dimensions, file, output_filename]
        else:
            specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_file, '--interpolation', self.interpolation_dict[self.interpolation], file, output_filename]
        
        subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

class Coregister(Preprocessor):

    def load(self, kwargs):

        add_parameter(self, kwargs, 'command', ['Slicer', '--launch', 'BRAINSFit'])

        add_parameter(self, kwargs, 'transform_type', 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine')
        add_parameter(self, kwargs, 'transform_initialization', 'useMomentsAlign')
        add_parameter(self, kwargs, 'interpolation', 'Linear')
        add_parameter(self, kwargs, 'sampling_percentage', .06)

        add_parameter(self, kwargs, 'reference_channel', None)
        add_parameter(self, kwargs, 'reference_file', None)

        add_parameter(self, kwargs, 'preprocessor_string', '_Registered')

        self.interpolation_dict = {'nearestNeighbor': 'nn'}

    def preprocess():

        if self.reference_channel is not None:
            if self.reference_channel == index:
                data_group.preprocessed_case[index] = file
                return True
            else:
                specific_command = self.command + ['--fixedVolume', data_group.preprocessed_case[self.reference_channel], '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', file, '--outputVolume', output_filename]
        else:
            specific_command = self.command + ['--fixedVolume', '"' + self.reference_file + '"', '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', file, '--outputVolume', output_filename]

        subprocess.call(' '.join(specific_command), shell=True)