""" A list of preprocessors that generally deal with geometric
    transformations of input data.
"""

import subprocess
import os
import numpy as np
import itertools

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.utilities.util import add_parameter, quotes

FNULL = open(os.devnull, 'w')


class ReorderAxes(Preprocessor):

    """ Equivalent to NumPy's transpose function. Reorders the axes in
        your data according to the axis_ordering parameter.

        axis_ordering: list or tuple
            New ordering of axes, in list format.
    """

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'ReorderAxes')

        # Dropping Parameters
        add_parameter(self, kwargs, 'axis_ordering', None)

        assert self.axis_ordering is not None, 'You must provide an axis ordering for ReorderAxes.'

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(ReorderAxes, self).initialize(data_collection)

        for label, data_group in list(self.data_groups.items()):

            # print(self.output_shape[label])
            self.output_shape[label] = self.axis_ordering

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case
        self.output_data = np.transpose(input_data, axes=self.axis_ordering)

        data_group.preprocessed_case = self.output_data


class SqueezeAxes(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'SqueezeAxes')

        # Dropping Parameters
        add_parameter(self, kwargs, 'axes', None)

        if type(self.axes) is not list and self.axes is not None:
            self.axes = [self.axes]

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(SqueezeAxes, self).initialize(data_collection)

        for label, data_group in list(self.data_groups.items()):

            data_shape = list(data_group.get_shape())
            
            # Messy, revise.
            new_shape = []
            if self.axes is None:
                for axis in data_shape:
                    if axis != 1:
                        new_shape += [axis]
            else:
                for axis in data_shape:
                    if axis != 1 and axis in self.axes:
                        new_shape += [axis]

            self.output_shape[label] = new_shape

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case
        output_data = np.squeeze(input_data, axis=self.axes)

        data_group.preprocessed_case = output_data
        self.output_data = output_data


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

        for label, data_group in list(self.data_groups.items()):
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


class OneHotEncode(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'OneHotEncode')

        # Class Parameters
        add_parameter(self, kwargs, 'num_classes', 3)
        add_parameter(self, kwargs, 'input_classes', None)
        add_parameter(self, kwargs, 'class_dictionary', {})

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(OneHotEncode, self).initialize(data_collection)

        if self.class_dictionary == {} and self.input_classes is not None:
            for idx, class_name in enumerate(self.input_classes):
                self.class_dictionary[class_name] = idx

        for label, data_group in list(self.data_groups.items()):
            data_shape = list(data_group.get_shape())
            data_shape[-1] = self.num_classes
            self.output_shape[label] = tuple(data_shape)

    def preprocess(self, data_group):

        # Relatively brittle, only works for 1-dimensional data.
        input_data = data_group.preprocessed_case

        # Probably not the most efficient.
        output_data = np.zeros(self.num_classes)
        for item in input_data:
            if self.class_dictionary != {}:
                output_data[self.class_dictionary[item]] = 1
            else:
                output_data[int(item)] = 1

        data_group.preprocessed_case = output_data
        self.output_data = output_data


class CopyChannels(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'CopyChannels')

        # Class Parameters
        add_parameter(self, kwargs, 'channel_multiplier', 3)
        add_parameter(self, kwargs, 'new_channel_dim', False)

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(CopyChannels, self).initialize(data_collection)

        for label, data_group in list(self.data_groups.items()):
            data_shape = list(data_group.get_shape())

            if self.new_channel_dim:
                data_shape += [self.channel_multiplier]
            else:
                data_shape[-1] = self.channel_multiplier * data_shape[-1]

            self.output_shape[label] = tuple(data_shape)

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case

        if self.new_channel_dim:
            output_data = np.tile(input_data[..., np.newaxis], (1, 1, self.channel_multiplier))
        else:
            output_data = np.tile(input_data, (1, 1, self.channel_multiplier))

        data_group.preprocessed_case = output_data
        self.output_data = output_data


class SelectChannels(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Merge')

        # Dropping Parameters
        add_parameter(self, kwargs, 'channels', [0, 1, 2, 3])

        self.output_shape = {}
        self.array_input = True

    def initialize(self, data_collection):

        super(SelectChannels, self).initialize(data_collection)

        for label, data_group in list(self.data_groups.items()):
            data_shape = list(data_group.get_shape())
            data_shape[-1] = len(self.channels)
            self.output_shape[label] = data_shape

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case
        output_data = np.take(input_data, self.channels, axis=-1)

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

        # If input data is different shapes, providing an output shape
        # does not make too much sense here. Common problem in all
        # DeepNeuro
        for label, data_group in list(self.data_groups.items()):
            data_shape = list(data_group.get_shape())
            data_shape[-1] = len(self.label_splits)
            self.output_shape[label] = data_shape

    def preprocess(self, data_group):

        """ I think there should be a more pythonic/numpythonic way to do this.
        """

        input_data = data_group.preprocessed_case
        output_shape = list(input_data.shape)
        output_shape[-1] = len(self.label_splits)
        output_data = np.zeros(output_shape)

        # Merge Target Channels
        if self.split_method == 'integer_levels':
            for label_idx, label in enumerate(self.label_splits):
                if type(label) is list:
                    # This is a little clunky
                    single_label_data = np.zeros(output_shape[0:-1])[..., np.newaxis]
                    for index in label:
                        single_label_data += np.where(input_data == index, 1, 0)
                    single_label_data = np.where(single_label_data > 0, 1, 0)
                else:
                    single_label_data = np.where(input_data == label, 1, 0)

                output_data[..., label_idx] = single_label_data[..., 0]

        data_group.preprocessed_case = output_data
        self.output_data = output_data


class CropValues(Preprocessor):

    """ Implemented from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        Removes all rows/columns/etc composed only of the provided values.

        mask_value: int
            Value to be cropped. Currently, only 0 supported.
        lead_data_group: str
            If provided, multiple data groups will be cropped according
            to the bounding box determined in the lead_data_group. Only
            applicable if different data_groups have the same shape. None
            by default, meaning that all data groups are cropped on an
            individual basis.
    """

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'CropValues')

        # Dropping Parameters
        add_parameter(self, kwargs, 'mask_value', 0)
        add_parameter(self, kwargs, 'lead_data_group', None)

        assert self.mask_value == 0, 'Nonzero mask_value has not yet been implemented.'

        self.slice_boundaries = []
        self.array_input = True

    def initialize(self, data_collection):

        super(CropValues, self).initialize(data_collection)

        if self.lead_data_group is not None:
            # Complex code to bring the lead data group to the front.
            lead_idx = None
            for idx, value in enumerate(self.data_groups_iterator):
                if value[0] == self.lead_data_group:
                    lead_idx = idx

            self.data_groups_iterator.insert(0, self.data_groups_iterator.pop(lead_idx))

        return

    def execute(self, data_collection, return_array=False):

        super(CropValues, self).execute(data_collection, return_array)

        self.slice_boundaries = []

    def preprocess(self, data_group):

        input_data = data_group.preprocessed_case

        if self.lead_data_group is None or self.slice_boundaries == []:
            num_dims = input_data.ndim
            slice_boundaries = []
            for axis in itertools.combinations(range(num_dims), num_dims - 1):
                nonzero = np.any(input_data, axis=axis)
                slice_boundary = list(np.where(nonzero)[0][[0, -1]])
                slice_boundaries = [slice(slice_boundary[0], slice_boundary[1] + 1, 1)] + slice_boundaries
                self.output_data = input_data[tuple(slice_boundaries)]
        else:
            self.output_data = input_data[tuple(self.slice_boundaries)]

        if self.slice_boundaries == [] and self.lead_data_group is not None:
            self.slice_boundaries = tuple(slice_boundaries)

        data_group.preprocessed_case = self.output_data


class Resize(Preprocessor):

    def load(self, kwargs):

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Resize')

        # Registration Parameters
        add_parameter(self, kwargs, 'output_shape', [1, 1, 1])
        add_parameter(self, kwargs, 'interpolation', 'linear')

        # Derived Parameters
        add_parameter(self, kwargs, 'preprocessor_string', '_Resized_' + str(self.output_shape).strip('[]').replace(' ', '').replace(',', ''))
        self.interpolation_dict = {'nearestNeighbor': 'nn', 'linear': 'linear'}
        self.dimensions = str(self.dimensions).strip('[]').replace(' ', '')

        self.array_input = True

    def preprocess(self, data_group):

        for file_idx, filename in enumerate(data_group.preprocessed_case):
            if self.reference_channel is None:
                specific_command = self.command + ['ResampleScalarVolume', '-i', self.interpolation, '-s', self.dimensions, quotes(filename), quotes(self.output_filenames[file_idx])]
            else:
                specific_command = self.command + ['ResampleScalarVectorDWIVolume', '-R', self.reference_channel, '--interpolation', self.interpolation_dict[self.interpolation], quotes(self.base_file), quotes(self.output_filename)]
            subprocess.call(' '.join(specific_command), shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

        self.output_data = self.output_filenames
        data_group.preprocessed_case = self.output_filenames


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
        add_parameter(self, kwargs, 'reference_channel', 0)
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