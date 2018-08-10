import subprocess
import os

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, quotes

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
                specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, '-s', self.dimensions, quotes(filename), quotes(self.output_filenames[file_idx])]
            else:
                specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_channel, '--interpolation', self.interpolation_dict[self.interpolation], quotes(self.base_file), quotes(self.output_filename)]
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

        # Output Parameters
        add_parameter(self, kwargs, 'save_transforms', True)
        add_parameter(self, kwargs, 'transform_string', '_coreg_transform')

        self.interpolation_dict = {'nearestNeighbor': 'nn'}
        self.array_input = False

    def initialize(self, data_collection):

        super(Coregister, self).initialize(data_collection)

        self.output_transforms = []

    def preprocess(self, data_group):

        for file_idx, filename in enumerate(data_group.preprocessed_case):

            transform_filename = self.generate_output_filename(filename, self.transform_string, file_extension='.tfm')

            if self.reference_channel is not None:
                if self.reference_channel == file_idx:
                    self.output_filenames[file_idx] = filename
                    self.output_transforms += [None]
                    continue
                else:
                    specific_command = self.command + ['--fixedVolume', quotes(data_group.preprocessed_case[self.reference_channel]), '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', quotes(filename), '--outputVolume', quotes(self.output_filenames[file_idx]), '--outputTransform', quotes(transform_filename)]
            else:
                specific_command = self.command + ['--fixedVolume', quotes(self.reference_file), '--transformType', self.transform_type, '--initializeTransformMode', self.transform_initialization, '--interpolationMode', self.interpolation, '--samplingPercentage', str(self.sampling_percentage), '--movingVolume', quotes(self.base_file), '--outputVolume', quotes(self.output_filename), '--outputTransform', quotes(transform_filename)]

            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

            # self.output_transforms += [read_image_files(transform_filename)]

            if not self.save_transforms:
                os.remove(transform_filename)

        self.output_data = self.output_filenames
        data_group.preprocessed_case = self.output_filenames

    def store_outputs(self, data_collection, data_group):

        self.data_dictionary[data_group.label]['output_transforms'] = self.output_transforms

        return super(Coregister, self).store_outputs(data_collection, data_group)