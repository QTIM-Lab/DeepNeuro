""" TODO: Break out these augmentations into submodules for easier reference.
    TODO: Rewrite this code to be briefer. Take advantage of common python class structures
"""

import numpy as np

from deepneuro.utilities.util import add_parameter


class Augmentation(object):

    def __init__(self, **kwargs):

        # Instance Options
        add_parameter(self, kwargs, 'data_groups', [])

        # Repetition Options
        add_parameter(self, kwargs, 'multiplier', None)
        add_parameter(self, kwargs, 'total', None)

        # Derived Parameters
        self.output_shape = None
        self.initialization = False
        self.iteration = 0
        self.data_groups = {data_group: None for data_group in self.data_groups}

        self.augmentation_string = '_copy_'
        self.load(kwargs)

        return

    def load(self, kwargs):

        return

    def set_multiplier(self, multiplier):

        self.multiplier = multiplier

    def augment(self, augmentation_num=0):

        for label, data_group in list(self.data_groups.items()):

            data_group.augmentation_cases[augmentation_num + 1] = data_group.augmentation_cases[augmentation_num]

    def initialize_augmentation(self):

        if not self.initialization:
            self.initialization = True

    def iterate(self):

        if self.multiplier is None:
            return

        self.iteration += 1
        if self.iteration == self.multiplier:
            self.iteration = 0

    def reset(self, augmentation_num):
        return

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group


# Aliasing
Copy = Augmentation


class Flip_Rotate_2D(Augmentation):

    """ TODO: extend to be more flexible and useful.
        Ponder about how best to apply to multiple dimensions
    """

    def load(self, kwargs):

        # Flip and Rotate options
        add_parameter(self, kwargs, 'flip', True)
        add_parameter(self, kwargs, 'rotate', True)
        add_parameter(self, kwargs, 'flip_axis', 2)
        add_parameter(self, kwargs, 'rotate_axis', (1, 2))
        add_parameter(self, kwargs, 'shuffle', True)

        # TODO: This is incredibly over-elaborate, return to fix.
        self.transforms_list = []

        if self.flip:
            self.flip_list = [False, True]
        else:
            self.flip_list = [False]

        if self.rotate:
            self.rotations_90 = [0, 1, 2, 3]
        else:
            self.rotations_90 = [0]

        self.available_transforms = np.array(np.meshgrid(self.flip_list, self.rotations_90)).T.reshape(-1, 2)
        self.total_transforms = self.available_transforms.shape[0]
        self.augmentation_string = '_rotate2D_'

    def reset(self, augmentation_num):

        if self.shuffle:
            np.random.shuffle(self.available_transforms)

    def augment(self, augmentation_num=0):

        for label, data_group in list(self.data_groups.items()):

            input_data = data_group.augmentation_cases[augmentation_num]

            if self.available_transforms[self.iteration % self.total_transforms, 0]:
                data_group.augmentation_cases[augmentation_num + 1] = np.flip(input_data, self.flip_axis)
                input_data = data_group.augmentation_cases[augmentation_num + 1]
            else:
                data_group.augmentation_cases[augmentation_num + 1] = input_data

            if self.available_transforms[self.iteration % self.total_transforms, 1]:
                data_group.augmentation_cases[augmentation_num + 1] = np.rot90(input_data, self.available_transforms[self.iteration % self.total_transforms, 1], axes=self.rotate_axis)


class Shift_Squeeze_Intensities(Augmentation):

    """ TODO: extend to be more flexible and useful.
        Ponder about how best to apply to multiple dimensions
    """

    def load(self, kwargs):

        # Flip and Rotate options
        add_parameter(self, kwargs, 'shift', True)
        add_parameter(self, kwargs, 'squeeze', True)
        add_parameter(self, kwargs, 'shift_amount', [-.5, .5])
        add_parameter(self, kwargs, 'squeeze_factor', [.7, 1.3])

        # TODO: This is incredibly over-elaborate, return to fix.
        self.transforms_list = []

        if self.shift:
            self.shift_list = [False, True]
        else:
            self.shift_list = [False]

        if self.squeeze:
            self.squeeze_list = [False, True]
        else:
            self.squeeze_list = [False]

        self.available_transforms = np.array(np.meshgrid(self.shift_list, self.squeeze_list)).T.reshape(-1, 2)
        self.total_transforms = self.available_transforms.shape[0]
        self.augmentation_string = '_shift_squeeze_'

    def augment(self, augmentation_num=0):

        for label, data_group in list(self.data_groups.items()):

            if self.available_transforms[self.iteration % self.total_transforms, 0]:
                data_group.augmentation_cases[augmentation_num + 1] = data_group.augmentation_cases[augmentation_num] + np.random.uniform(self.shift_amount[0], self.shift_amount[1])
            else:
                data_group.augmentation_cases[augmentation_num + 1] = data_group.augmentation_cases[augmentation_num]

            if self.available_transforms[self.iteration % self.total_transforms, 0]:
                data_group.augmentation_cases[augmentation_num + 1] = data_group.augmentation_cases[augmentation_num] * np.random.uniform(self.squeeze_factor[0], self.squeeze_factor[1])
            else:
                data_group.augmentation_cases[augmentation_num + 1] = data_group.augmentation_cases[augmentation_num]


class Flip_Rotate_3D(Augmentation):

    def load(self, kwargs):

        """
        """

        # Flip and Rotate options
        add_parameter(self, kwargs, 'rotation_axes', [1, 2, 3])

        # Derived Parameters
        self.rotation_generator = {}
        self.augmentation_num = 0

    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in list(self.data_groups.items()):
                self.rotation_generator[label] = self.rotations24(data_group.augmentation_cases[0])

            self.initialization = True

    def rotations24(self, array):

        while True:
            # imagine shape is pointing in axis 0 (up)

            # 4 rotations about axis 0
            for i in range(4):
                yield self.rot90_3d(array, i, self.rotation_axes[0])

            # rotate 180 about axis 1, now shape is pointing down in axis 0
            # 4 rotations about axis 0
            rotated_array = self.rot90_3d(array, 2, axis=self.rotation_axes[1])
            for i in range(4):
                yield self.rot90_3d(rotated_array, i, self.rotation_axes[0])

            # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
            # 8 rotations about axis 2
            rotated_array = self.rot90_3d(array, axis=self.rotation_axes[1])
            for i in range(4):
                yield self.rot90_3d(rotated_array, i, self.rotation_axes[2])

            rotated_array = self.rot90_3d(array, -1, axis=self.rotation_axes[1])
            for i in range(4):
                yield self.rot90_3d(rotated_array, i, self.rotation_axes[2])

            # rotate about axis 2, now shape is pointing in axis 1
            # 8 rotations about axis 1
            rotated_array = self.rot90_3d(array, axis=self.rotation_axes[2])
            for i in range(4):
                yield self.rot90_3d(rotated_array, i, self.rotation_axes[1])

            rotated_array = self.rot90_3d(array, -1, axis=self.rotation_axes[2])
            for i in range(4):
                yield self.rot90_3d(rotated_array, i, self.rotation_axes[1])

    def rot90_3d(self, m, k=1, axis=2):
        """Rotate an array by 90 degrees in the counter-clockwise direction around the given axis"""
        m = np.swapaxes(m, 2, axis)
        m = np.rot90(m, k)
        m = np.swapaxes(m, 2, axis)
        return m

    def augment(self, augmentation_num=0):

        # Hacky -- the rotation generator is weird here.
        if augmentation_num != self.augmentation_num:
            self.augmentation_num = augmentation_num
        for label, data_group in list(self.data_groups.items()):
            self.rotation_generator[label] = self.rotations24(data_group.augmentation_cases[self.augmentation_num])

        for label, data_group in list(self.data_groups.items()):

            data_group.augmentation_cases[augmentation_num + 1] = next(self.rotation_generator[label])


class MaskData(Augmentation):

    def load(self, kwargs):

        # Add functionality for masking multiples axes.

        # Mask Parameters
        add_parameter(self, kwargs, 'mask_channels', {})
        add_parameter(self, kwargs, 'num_masked', 1)
        add_parameter(self, kwargs, 'masked_value', -10)
        add_parameter(self, kwargs, 'random_sample', True)

        # Derived Parameters
        self.input_shape = {}
        self.augmentation_string = '_mask_'

    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in list(self.data_groups.items()):
                self.mask_channels[label] = np.array(self.mask_channels[label])
                # self.input_shape[label] = data_group.get_shape()
                # if label not in self.mask_channels.keys():
                    # self.mask_channels[label] = np.arange(self.input_shape[label][-1])
                # else:
                    # self.mask_channels[label] = np.arange(self.input_shape[label][self.mask_channels[label] + 1])

            self.initialization = True

    def iterate(self):

        super(MaskData, self).iterate()

    def augment(self, augmentation_num=0):

        for label, data_group in list(self.data_groups.items()):

            if self.random_sample:
                channels = np.random.choice(self.mask_channels[label], self.num_masked, replace=False)
            else:
                idx = [x % len(self.mask_channels[label]) for x in range(self.iteration, self.iteration + self.num_masked)]
                channels = self.mask_channels[label][idx]

            # Currently only works if applied to channels; revisit
            masked_data = np.copy(data_group.augmentation_cases[augmentation_num])

            # for channel in channels:
            masked_data[..., channels] = self.masked_value
            
            data_group.augmentation_cases[augmentation_num + 1] = masked_data
            data_group.augmentation_strings[augmentation_num + 1] = data_group.augmentation_strings[augmentation_num] + self.augmentation_string + str(channels).strip('[]').replace(' ', '')


class Downsample(Augmentation):

    def load(self, kwargs):

        # A lot of this functionality is vague and messy, revisit

        # Downsample Parameters
        add_parameter(self, kwargs, 'channel', 0)
        add_parameter(self, kwargs, 'axes', {})
        add_parameter(self, kwargs, 'factor', 2)
        add_parameter(self, kwargs, 'random_sample', True)
        add_parameter(self, kwargs, 'num_downsampled', 1)

        self.input_shape = {}
        self.augmentation_string = '_resample_'

    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in list(self.data_groups.items()):
                self.input_shape[label] = data_group.get_shape()

            self.initialization = True

    def iterate(self):

        super(Downsample, self).iterate()

    def augment(self, augmentation_num=0):

        for label, data_group in list(self.data_groups.items()):

            if self.random_sample:
                axes = np.random.choice(self.axes[label], self.num_downsampled, replace=False)
            else:
                idx = [x % len(self.axes[label]) for x in range(self.iteration, self.iteration + self.num_downsampled)]
                axes = np.array(self.axes[label])[idx]

            resampled_data = np.copy(data_group.augmentation_cases[augmentation_num])

            # TODO: Put in utlitities for different amount of resampling in different dimensions
            # This is fun, but messy, rewrite
            static_slice = [slice(None)] * (len(resampled_data.shape) - 1) + [slice(self.channel, self.channel + 1)]
            for axis in axes:
                static_slice[axis] = slice(0, None, self.factor)

            replaced_slices = [[slice(None)] * (len(resampled_data.shape) - 1) + [slice(self.channel, self.channel + 1)] for i in range(self.factor - 1)]
            for axis in axes:
                for i in range(self.factor - 1):
                    replaced_slices[i][axis] = slice(i + 1, None, self.factor)

            # Gross. Would be nice to find an elegant/effecient way to do this.
            for duplicate in replaced_slices:
                replaced_data = resampled_data[duplicate]
                replacing_data = resampled_data[static_slice]
                # Need geometric axes to reference here, currently non-functional in most cases.
                for axis in [-4, -3, -2]:
                    if replaced_data.shape[axis] < replacing_data.shape[axis]:
                        hedge_slice = [slice(None)] * (replacing_data.ndim)
                        hedge_slice[axis] = slice(0, replaced_data.shape[axis])
                        replacing_data = replacing_data[hedge_slice]

                resampled_data[duplicate] = replacing_data

            data_group.augmentation_cases[augmentation_num + 1] = resampled_data           
            data_group.augmentation_strings[augmentation_num + 1] = data_group.augmentation_strings[augmentation_num] + self.augmentation_string + str(self.factor) + '_' + str(axes).strip('[]').replace(' ', '')


if __name__ == '__main__':
    pass