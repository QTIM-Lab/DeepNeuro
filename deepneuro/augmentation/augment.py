""" TODO: Break out these augmentations into submodules for easier reference.
    TODO: Rewrite this code to be briefer. Take advantage of common python class structures
"""

import numpy as np

from scipy.sparse import csr_matrix

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

        for label, data_group in self.data_groups.items():

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
        self.augmentation_string = '_roate3D_'

    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in self.data_groups.items():
                # Dealing with the time dimension.
                if len(data_group.get_shape()) < 5:
                    self.flip_axis = 1
                else:
                    self.flip_axis = -4

            self.initialization = True

    def augment(self, augmentation_num=0):

        for label, data_group in self.data_groups.items():

            if self.available_transforms[self.iteration % self.total_transforms, 0]:
                data_group.augmentation_cases[augmentation_num + 1] = np.flip(data_group.augmentation_cases[augmentation_num], self.flip_axis)
            else:
                data_group.augmentation_cases[augmentation_num + 1] = data_group.augmentation_cases[augmentation_num]

            if self.available_transforms[self.iteration % self.total_transforms, 1]:
                data_group.augmentation_cases[augmentation_num + 1] = np.rot90(data_group.augmentation_cases[augmentation_num], self.available_transforms[self.iteration % self.total_transforms, self.flip_axis])


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

        for label, data_group in self.data_groups.items():

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

            for label, data_group in self.data_groups.items():
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
        for label, data_group in self.data_groups.items():
            self.rotation_generator[label] = self.rotations24(data_group.augmentation_cases[self.augmentation_num])

        for label, data_group in self.data_groups.items():

            data_group.augmentation_cases[augmentation_num + 1] = next(self.rotation_generator[label])


class ExtractPatches(Augmentation):

    def load(self, kwargs):

        # Patch Parameters
        add_parameter(self, kwargs, 'patch_shape', None)
        add_parameter(self, kwargs, 'patch_extraction_conditions', None)
        add_parameter(self, kwargs, 'patch_region_conditions', None)
        add_parameter(self, kwargs, 'patch_dimensions', {})

        # Derived Parameters
        self.patch_regions = []
        self.patches = None
        self.patch_corner = None
        self.patch_slice = None
        self.leading_dims = {}
        self.input_shape = {}
        self.output_shape = {}  # Redundant
        self.augmentation_string = '_patch_'

    def initialize_augmentation(self):

        """ There are some batch dimension problems with output_shape here. Hacky fixes for now, but revisit. TODO
        """

        if not self.initialization:

            # A weird way to proportionally divvy up patch conditions.
            # TODO: Rewrite
            self.condition_list = [None] * (self.multiplier)
            self.region_list = [None] * (self.multiplier)

            if self.patch_extraction_conditions is not None:
                start_idx = 0
                for condition_idx, patch_extraction_condition in enumerate(self.patch_extraction_conditions):
                    end_idx = start_idx + int(np.ceil(patch_extraction_condition[1] * self.multiplier))
                    self.condition_list[start_idx:end_idx] = [condition_idx] * (end_idx - start_idx)
                    start_idx = end_idx

            if self.patch_region_conditions is not None:
                start_idx = 0
                for condition_idx, patch_region_condition in enumerate(self.patch_region_conditions):
                    end_idx = start_idx + int(np.ceil(patch_region_condition[1] * self.multiplier))
                    self.region_list[start_idx:end_idx] = [condition_idx] * (end_idx - start_idx)
                    start_idx = end_idx

            for label, data_group in self.data_groups.items():
                self.input_shape[label] = data_group.get_shape()
                if label not in list(self.patch_dimensions.keys()):
                    # If no provided patch dimensions, just presume the format is [batch, patch_dimensions, channel]
                    # self.patch_dimensions[label] = [-4 + x for x in xrange(len(self.input_shape[label]) - 1)]
                    self.patch_dimensions[label] = [x + 1 for x in range(len(self.input_shape[label]) - 1)]

                # This is a little goofy.
                self.output_shape[label] = np.array(self.input_shape[label])
                # self.output_shape[label][self.patch_dimensions[label]] = list(self.patch_shape)
                self.output_shape[label][[x - 1 for x in self.patch_dimensions[label]]] = list(self.patch_shape)
                self.output_shape[label] = tuple(self.output_shape[label])

                # Batch dimension correction, revisit
                # self.patch_dimensions[label] = [x + 1 for x in self.patch_dimensions[label]]

            self.initialization = True

    def iterate(self):

        super(ExtractPatches, self).iterate()

        self.generate_patch_corner()

    def reset(self, augmentation_num=0):

        self.patch_regions = []
        region_input_data = {label: self.data_groups[label].augmentation_cases[augmentation_num] for label in list(self.data_groups.keys())}
        for region_condition in self.patch_region_conditions:
            # print 'Extracting region for..', region_condition
            # self.patch_regions += [np.where(region_condition[0](region_input_data))]
            self.patch_regions += self.get_indices_sparse(region_condition[0](region_input_data))

        return

    def augment(self, augmentation_num=0):

        # Any more sensible way to deal with this case?
        if self.patches is None:
            self.generate_patch_corner(augmentation_num)

        for label, data_group in self.data_groups.items():

            # A bit lengthy. Also unnecessarily rebuffers patches
            data_group.augmentation_cases[augmentation_num + 1] = self.patches[label]

    def generate_patch_corner(self, augmentation_num=0):

        """ Think about how one could to this, say, with 3D and 4D volumes at the same time.
            Also, patching across the modality dimension..? Interesting..
        """

        # TODO: Escape clause in case acceptable patches cannot be found.

        # acceptable_patch = False

        region = self.patch_regions[self.region_list[self.iteration]]

        # TODO: Make errors like these more ubiquitous.
        if len(region[0]) == 0:
            # raise ValueError('The region ' + str(self.patch_region_conditions[self.region_list[self.iteration]][0]) + ' has no voxels to select patches from. Please modify your patch-sampling region')
            # Tempfix -- Eek
            region = self.patch_regions[self.region_list[1]]
        if len(region[0]) == 0:
            print('emergency brain region..')
            region = np.where(self.data_groups['input_modalities'].augmentation_cases[augmentation_num] != 0)
            self.patch_regions[self.region_list[0]] = region
        
        corner_idx = np.random.randint(len(region[0]))

        self.patches = {}

        # Pad edge patches.
        for label, data_group in self.data_groups.items():

            # TODO: Some redundancy here
            corner = np.array([d[corner_idx] for d in region])[self.patch_dimensions[label]]

            patch_slice = [slice(None)] * (len(self.input_shape[label]) + 1)
            # Will run into problems with odd-shaped patches.
            for idx, patch_dim in enumerate(self.patch_dimensions[label]):
                patch_slice[patch_dim] = slice(max(0, corner[idx] - self.patch_shape[idx] / 2), corner[idx] + self.patch_shape[idx] / 2, 1)

            input_shape = self.data_groups[label].augmentation_cases[augmentation_num].shape

            self.patches[label] = self.data_groups[label].augmentation_cases[augmentation_num][patch_slice]

            # More complicated padding needed for center-voxel based patches.
            pad_dims = [(0, 0)] * len(self.patches[label].shape)
            for idx, patch_dim in enumerate(self.patch_dimensions[label]):
                pad = [0, 0]
                if corner[idx] > input_shape[patch_dim] - self.patch_shape[idx] / 2:
                    pad[1] = self.patch_shape[idx] / 2 - (input_shape[patch_dim] - corner[idx])
                if corner[idx] < self.patch_shape[idx] / 2:
                    pad[0] = self.patch_shape[idx] / 2 - corner[idx]
                pad_dims[patch_dim] = tuple(pad)

            # print 'label', label
            # print 'input_shape', self.data_groups[label].augmentation_cases[augmentation_num].shape
            # print 'patch_slice', patch_slice
            # print 'pad_dims', pad_dims
            # print self.patches[label].shape
            self.patches[label] = np.lib.pad(self.patches[label], tuple(pad_dims), 'edge')
            # print self.patches[label].shape, 'post-pad'

        return

    def compute_M(self, data):

        # Magic, vectorized sparse matrix calculation method to replace np.where
        # https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where

        cols = np.arange(data.size)
        return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))

    def get_indices_sparse(self, data):

        # Magic, vectorized sparse matrix calculation method to replace np.where
        # https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where

        M = self.compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]


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

            for label, data_group in self.data_groups.items():
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

        for label, data_group in self.data_groups.items():

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


class ChooseData(Augmentation):

    def load(self, kwargs):

        # Add functionality for choosing multiple axes

        # Choose Parameters
        add_parameter(self, kwargs, 'axis', {})
        add_parameter(self, kwargs, 'choices', None)
        add_parameter(self, kwargs, 'num_chosen', 1)
        add_parameter(self, kwargs, 'random_sample', True)

        # Derived Parameters
        self.input_shape = {}
        self.augmentation_string = '_choose_'

    def initialize_augmentation(self):

        if not self.initialization:

            self.choices = np.array(self.choices)

            for label, data_group in self.data_groups.items():
                input_shape = data_group.get_shape()
                self.output_shape[label] = np.array(input_shape)
                self.output_shape[label][self.axis[label]] = self.num_chosen
                self.output_shape[label] = tuple(self.output_shape[label])

            self.initialization = True

    def iterate(self):

        super(ChooseData, self).iterate()

    def augment(self, augmentation_num=0):

        choice = None  # This is messed up

        for label, data_group in self.data_groups.items():

            # Wrote this function while half-asleep; revisit
            input_data = data_group.augmentation_cases[augmentation_num]

            if self.choices is None:
                choices = np.arange(input_data.shape[self.axis[label]])
            else:
                choices = self.choices

            if choice is None:
                if self.random_sample:
                    choice = np.random.choice(choices, self.num_chosen, replace=False)
                else:
                    idx = [x % len(choices) for x in range(self.iteration, self.iteration + self.num_chosen)]
                    choice = choices[idx]

            # Temporary
            if input_data.shape[-1] == 6:
                choice = choice.tolist()
                choice = list(range(4)) + choice

            choice_slice = [slice(None)] * (len(input_data.shape))
            choice_slice[self.axis[label]] = choice

            # Currently only works if applied to channels; revisit
            data_group.augmentation_cases[augmentation_num + 1] = input_data[choice_slice]
            data_group.augmentation_strings[augmentation_num + 1] = data_group.augmentation_strings[augmentation_num] + self.augmentation_string + str(choice).strip('[]').replace(' ', '')


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

            for label, data_group in self.data_groups.items():
                self.input_shape[label] = data_group.get_shape()

            self.initialization = True

    def iterate(self):

        super(Downsample, self).iterate()

    def augment(self, augmentation_num=0):

        for label, data_group in self.data_groups.items():

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