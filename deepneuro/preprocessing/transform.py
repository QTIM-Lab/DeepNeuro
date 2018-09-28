import subprocess
import os
import numpy as np

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, quotes

FNULL = open(os.devnull, 'w')


class MergeChannels(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Merge')

        # Merge Parameters
        add_parameter(self, kwargs, 'channels', None)
        add_parameter(self, kwargs, 'merge_method', 'maximum')

        self.output_shape = {}

        self.array_input = True

    def initialize(self, data_collection):

        super(MergeChannels, self).initialize(data_collection)

        if self.channels is None:
            self.output_num = 1
        else:
            self.output_num = len(self.channels)

        for label, data_group in self.data_groups.items():
            data_shape = list(data_group.get_shape())
            if self.channels is None:
                data_shape[-1] = 1
            else:
                data_shape[-1] = data_shape[-1] - len(self.channels) + 1
            self.output_shape[label] = data_shape

    def preprocess(self, data_group):

        """ I think there should be a more pythonic/numpythonic way to do this.
        """

        input_data = data_group.preprocessed_case

        # Split Channels
        if self.channels is None:
            channel_subset = np.copy(input_data)
        else:
            all_channels = set(range(input_data.shape[-1]))
            remaining_channels = list(all_channels.difference(set(self.channels)))
            reminaing_channel_subset = np.take(input_data, remaining_channels, axis=-1)
            channel_subset = np.take(input_data, self.channels, axis=-1)

        # Merge Target Channels
        if self.merge_method == 'maximum':
            channel_subset = np.max(channel_subset, axis=-1)[..., np.newaxis]

        # Join Channels
        if self.channels is None:
            output_data = channel_subset
        else:
            output_data = np.concatenate((reminaing_channel_subset, channel_subset), axis=-1)

        data_group.preprocessed_case = output_data
        self.output_data = output_data


class SplitData(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Merge')

        # Splitting Parameters
        add_parameter(self, kwargs, 'split_method', 'integer_levels')
        add_parameter(self, kwargs, 'label_splits', [1, 2, 3, 4])

        self.output_shape = {}

        self.array_input = True

    def initialize(self, data_collection):

        super(SplitData, self).initialize(data_collection)

        for label, data_group in self.data_groups.items():
            data_shape = list(data_group.get_shape())
            data_shape[-1] = len(self.label_splits)
            self.output_shape[label] = data_shape

    def preprocess(self, data_group):

        """ I think there should be a more pythonic/numpythonic way to do this.
        """

        input_data = data_group.preprocessed_case
        output_data = np.zeros(self.output_shape[data_group.label])

        # Merge Target Channels
        if self.split_method == 'integer_levels':
            for label_idx, label in enumerate(self.label_splits):
                if type(label) is list:
                    # This is a little clunky
                    single_label_data = np.zeros(self.output_shape[data_group.label][0:-1])[..., np.newaxis]
                    for index in label:
                        single_label_data += np.where(input_data == index, 1, 0)
                    single_label_data = np.where(single_label_data > 0, 1, 0)
                else:
                    single_label_data = np.where(input_data == label, 1, 0)

                output_data[..., label_idx] = single_label_data[..., 0]

        data_group.preprocessed_case = output_data
        self.output_data = output_data


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