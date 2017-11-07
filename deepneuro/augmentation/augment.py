import numpy as np

class Augmentation(object):


    def __init__(self, data_groups=None, multiplier=None, total=None):

        # Note: total feature is as of yet unimplemented..

        self.multiplier = multiplier
        self.total = total

        self.output_shape = None
        self.initialization = False
        self.iteration = 0

        self.total_iterations = multiplier

        self.data_groups = {data_group: None for data_group in data_groups}

        return


    def set_multiplier(self, multiplier):

        self.multiplier = multiplier
        self.total_iterations = multiplier

    def augment(self, augmentation_num=0):

        for label, data_group in self.data_groups.iteritems():

            data_group.augmentation_cases[augmentation_num+1] = data_group.augmentation_cases[augmentation_num]

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

class Copy(Augmentation):

    def __init__(self, data_groups=None, multiplier=None, total=None):

        # Get rid of these with super??
        self.multiplier = multiplier
        self.total = total

        self.output_shape = None
        self.initialization = False
        self.iteration = 0

        self.total_iterations = multiplier

        self.data_groups = {data_group: None for data_group in data_groups}

class Flip_Rotate_2D(Augmentation):

    def __init__(self, data_groups=None, multiplier=None, total=None, flip=True, rotate=True):

        # Get rid of these with super??
        self.multiplier = multiplier
        self.total = total
        self.flip = flip
        self.rotate = rotate
        self.flip_axis = 1

        self.output_shape = None
        self.initialization = False
        self.iteration = 0

        self.total_iterations = multiplier

        self.data_groups = {data_group: None for data_group in data_groups}

        # TODO: This is incredibly over-elaborate, return to fix.
        self.transforms_list = []

        if self.flip:
            self.flip_list = [False, True]
        else:
            self.flip_list = [False]

        if self.rotate:
            self.rotations_90 = [0,1,2,3]
        else:
            self.rotations_90 = [0]

        self.available_transforms = np.array(np.meshgrid(self.flip_list, self.rotations_90)).T.reshape(-1,2)
        self.total_transforms = self.available_transforms.shape[0]


    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in self.data_groups.iteritems():
                # Dealing with the time dimension.
                if len(data_group.get_shape()) < 5:
                    self.flip_axis = 1
                else:
                    self.flip_axis = -4

            self.initialization = True

    def iterate(self):

        super(Flip_Rotate_2D, self).iterate()

    def augment(self, augmentation_num=0):

        print 'flippin'

        for label, data_group in self.data_groups.iteritems():

            if self.available_transforms[self.iteration % self.total_transforms, 0]:
                data_group.augmentation_cases[augmentation_num+1] = np.flip(data_group.augmentation_cases[augmentation_num], self.flip_axis)
            else:
                data_group.augmentation_cases[augmentation_num+1] = data_group.augmentation_cases[augmentation_num]

            if self.available_transforms[self.iteration % self.total_transforms, 1]:
                data_group.augmentation_cases[augmentation_num+1] = np.rot90(data_group.augmentation_cases[augmentation_num], self.available_transforms[self.iteration % self.total_transforms, self.flip_axis])

class ExtractPatches(Augmentation):

    def __init__(self, patch_shape, patch_region_conditions=None, patch_extraction_conditions=None, patch_dimensions={}, data_groups=None):

        # Get rid of these with super??
        self.multiplier = None
        self.total = None
        self.output_shape = None

        self.data_groups = {data_group: None for data_group in data_groups}

        self.initialization = False
        self.iteration = 0

        self.leading_dims = {}

        self.patch_shape = patch_shape
        self.patch_extraction_conditions = patch_extraction_conditions
        self.patch_region_conditions = patch_region_conditions
        self.patch_dimensions = patch_dimensions
        self.patch_regions = []
        self.patches = None
        self.patch_corner = None
        self.patch_slice = None

        self.input_shape = {}
        self.output_shape = {}

    def initialize_augmentation(self):

        if not self.initialization:

            # A weird way to proportionally divvy up patch conditions.
            # TODO: Rewrite
            self.condition_list = [None] * (self.multiplier)
            self.region_list = [None] * (self.multiplier)

            if self.patch_extraction_conditions is not None:
                start_idx = 0
                for condition_idx, patch_extraction_condition in enumerate(self.patch_extraction_conditions):
                    end_idx = start_idx + int(np.ceil(patch_extraction_condition[1]*self.multiplier))
                    self.condition_list[start_idx:end_idx] = [condition_idx]*(end_idx-start_idx)
                    start_idx = end_idx

            if self.patch_region_conditions is not None:
                start_idx = 0
                for condition_idx, patch_region_condition in enumerate(self.patch_region_conditions):
                    end_idx = start_idx + int(np.ceil(patch_region_condition[1]*self.multiplier))
                    self.region_list[start_idx:end_idx] = [condition_idx]*(end_idx-start_idx)
                    start_idx = end_idx

            for label, data_group in self.data_groups.iteritems():
                self.input_shape[label] = data_group.get_shape()
                if label not in self.patch_dimensions.keys():
                    # If no provided patch dimensions, just presume the format is [batch, patch_dimensions, channel]
                    self.patch_dimensions[label] = [-2 - x for x in xrange(len(self.input_shape[label]) - 1)]
                # This is a little goofy.
                self.output_shape[label] = np.array(self.input_shape[label])
                self.output_shape[label][self.patch_dimensions[label]] = list(self.patch_shape)
                self.output_shape[label] = tuple(self.output_shape[label])

            self.initialization = True

    def iterate(self):

        super(ExtractPatches, self).iterate()

        self.generate_patch_corner()

    def reset(self, augmentation_num=0):

        self.patch_regions = []
        region_input_data = {label: self.data_groups[label].augmentation_cases[augmentation_num] for label in self.data_groups.keys()}
        for region_condition in self.patch_region_conditions:
            print 'Extracting region for..', region_condition
            self.patch_regions += [np.where(region_condition[0](region_input_data))]

        return

    def augment(self, augmentation_num=0):

        # Any more sensible way to deal with this case?
        if self.patches is None:
            self.generate_patch_corner(augmentation_num)

        for label, data_group in self.data_groups.iteritems():

            # A bit lengthy. Also unnecessarily rebuffers patches
            data_group.augmentation_cases[augmentation_num + 1] = self.patches[label]

    def generate_patch_corner(self, augmentation_num=0):

        """ Think about how one could to this, say, with 3D and 4D volumes at the same time.
            Also, patching across the modality dimension..? Interesting..
        """

        # TODO: Escape clause in case acceptable patches cannot be found.

        acceptable_patch = False

        data_group_labels = self.data_groups.keys()

        # Hmm... How to make corner search process dimension-agnostic.
        leader_data_group = self.data_groups[data_group_labels[0]]
        data_shape = leader_data_group.augmentation_cases[augmentation_num].shape

        if self.patch_regions == []:
            # Currently non-functional, non-masked based sampling.
            while not acceptable_patch:

                corner = [np.random.randint(0, max_dim) for max_dim in data_shape[1:-1]]
                patch_slice = [slice(None)]*self.leading_dims + [slice(corner_dim, corner_dim+self.patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)] + [slice(None)]

                self.patches = {}

                # Pad edge patches.
                for key in self.data_groups:
                    self.patches[key] = self.data_groups[key].augmentation_cases[augmentation_num][patch_slice]

                    pad_dims = [(0,0)]
                    for idx, dim in enumerate(self.patches[key].shape[1:-1]):
                        pad_dims += [(0, self.patch_shape[idx]-dim)]
                    pad_dims += [(0,0)]

                    self.patches[key]  = np.lib.pad(self.patches[key] , tuple(pad_dims), 'edge')

                if self.condition_list[self.iteration] is not None:
                    acceptable_patch = self.patch_extraction_conditions[self.condition_list[self.iteration]][0](self.patches)
                else:
                    acceptable_patch = True
                    print self.patch_extraction_conditions[self.condition_list[self.iteration]][0]

        else:
            region = self.patch_regions[self.region_list[self.iteration]]

            # TODO: Make errors like these more ubiquitous.
            if len(region[0]) == 0:
                raise ValueError('The region ' + str(self.patch_region_conditions[self.region_list[self.iteration]][0]) + ' has no voxels to select patches from. Please modify your patch-sampling region')

            corner_idx = np.random.randint(len(region[0]))

            self.patches = {}

            # Pad edge patches.
            for label, data_group in self.data_groups.iteritems():

                # TODO: Some redundancy here
                corner = np.array([d[corner_idx] for d in region])[self.patch_dimensions[label]]

                patch_slice = [slice(None)] * len(self.input_shape[label])
                # Will run into problems with odd-shaped patches.
                for idx, patch_dim in enumerate(self.patch_dimensions[label]):
                    patch_slice[patch_dim] = slice(max(0, corner[idx] - self.patch_shape[idx]/2), corner[idx] + self.patch_shape[idx]/2, 1)
                
                input_shape = self.data_groups[label].augmentation_cases[augmentation_num].shape

                self.patches[label] = self.data_groups[label].augmentation_cases[augmentation_num][patch_slice]
                
                # More complicated padding needed for center-voxel based patches.
                pad_dims = [(0,0)] * len(self.patches[label].shape)
                for idx, patch_dim in enumerate(self.patch_dimensions[label]):
                    pad = [0,0]
                    if corner[idx] > input_shape[patch_dim] - self.patch_shape[idx]/2:
                        pad[1] = self.patch_shape[idx]/2 - (input_shape[patch_dim] - corner[idx])
                    if corner[idx] < self.patch_shape[idx]/2:
                        pad[0] = self.patch_shape[idx]/2 - corner[idx]
                    pad_dims[patch_dim] = tuple(pad)

                self.patches[label] = np.lib.pad(self.patches[label] , tuple(pad_dims), 'edge')

            print self.patch_region_conditions[self.region_list[self.iteration]][0]

        return

class MaskData(Augmentation):

    def __init__(self, data_groups=None, multiplier=None, total=None, mask_axis={}, num_masked=1, masked_value=-10, random_sample=True):

        # Get rid of these with super??
        self.multiplier = multiplier
        self.total = total

        # Add functionality for masking multiples axes.
        self.mask_axis = mask_axis
        self.num_masked = num_masked
        self.masked_value = -10
        self.random_sample = random_sample

        self.input_shape = {}

        self.output_shape = None
        self.initialization = False
        self.iteration = 0

        self.total_iterations = multiplier

        self.data_groups = {data_group: None for data_group in data_groups}
        self.augmentation_string = '_mask_'

    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in self.data_groups.iteritems():
                self.input_shape[label] = data_group.get_shape()
                if label not in self.mask_axis.keys():
                    self.mask_axis[label] = np.arange(self.input_shape[label][-1])
                else:
                    self.mask_axis[label] = np.arange(self.input_shape[label][self.mask_axis[label]])

            self.initialization = True

    def iterate(self):

        super(MaskData, self).iterate()

    def augment(self, augmentation_num=0):

        for label, data_group in self.data_groups.iteritems():

            if self.random_sample:
                channels = np.random.choice(self.mask_axis[label], self.num_masked, replace=False)
            else:
                idx = [x % len(self.mask_axis[label]) for x in xrange(self.iteration, self.iteration + self.num_masked)]
                channels = self.mask_axis[label][idx]

            # Currently only works if applied to channels; revisit
            masked_data = np.copy(data_group.augmentation_cases[augmentation_num])
            masked_data[...,channels] = self.masked_value
            data_group.augmentation_cases[augmentation_num+1] = masked_data
            data_group.augmentation_strings[augmentation_num+1] = data_group.augmentation_strings[augmentation_num] + self.augmentation_string + str(channels).strip('[]').replace(' ', '')

class Downsample(Augmentation):

    def __init__(self, data_groups=None, multiplier=None, total=None, channel=0, axes={}, factor=2, num_downsampled=1, random_sample=True):

        # Get rid of these with super??
        self.multiplier = multiplier
        self.total = total

        # A lot of this functionality is vague and messy, revisit
        self.channel = channel
        self.axes = axes
        self.factor = factor
        self.random_sample = random_sample
        self.num_downsampled = num_downsampled

        self.input_shape = {}

        self.output_shape = None
        self.initialization = False
        self.iteration = 0

        self.total_iterations = multiplier

        self.data_groups = {data_group: None for data_group in data_groups}
        self.augmentation_string = '_resample_'

        self.input_shape = {}


    def initialize_augmentation(self):

        if not self.initialization:

            for label, data_group in self.data_groups.iteritems():
                self.input_shape[label] = data_group.get_shape()

            self.initialization = True

    def iterate(self):

        super(Downsample, self).iterate()

    def augment(self, augmentation_num=0):

        for label, data_group in self.data_groups.iteritems():

            if self.random_sample:
                axes = np.random.choice(self.axes[label], self.num_downsampled, replace=False)
            else:
                idx = [x % len(self.axes[label]) for x in xrange(self.iteration, self.iteration + self.num_downsampled)]
                axes = self.axes[label][idx]

            resampled_data = np.copy(data_group.augmentation_cases[augmentation_num])

            # TODO: Put in utlitities for different amount of resampling in different dimensions
            # This is fun, but messy, rewrite
            static_slice = [slice(None)] * (len(self.input_shape[label]) - 1) + [slice(self.channel, self.channel + 1) ]
            for axis in axes:
                static_slice[axis] = slice(0, None, self.factor)

            replaced_slices = [[slice(None)] * (len(self.input_shape[label]) - 1) + [slice(self.channel, self.channel + 1)] for i in xrange(self.factor - 1)]
            for axis in axes:
                for i in xrange(self.factor - 1):
                    replaced_slices[i][axis] = slice(i+1, None, self.factor)

            # Gross. Would be nice to find an elegant/effecient way to do this.
            for duplicate in replaced_slices:
                replaced_data = resampled_data[duplicate]
                replacing_data = resampled_data[static_slice]
                for axis in axes:
                    if replaced_data.shape[axis] < replacing_data.shape[axis]:
                        hedge_slice = [slice(None)] * (len(self.input_shape[label]))
                        hedge_slice[axis] = slice(0, replaced_data.shape[axis])
                        replacing_data = replacing_data[hedge_slice]

                resampled_data[duplicate] = replacing_data
                
            data_group.augmentation_cases[augmentation_num+1] = resampled_data
            print data_group.augmentation_cases[augmentation_num].shape
            print resampled_data.shape
            data_group.augmentation_strings[augmentation_num+1] = data_group.augmentation_strings[augmentation_num] + self.augmentation_string + str(self.factor) + '_' + str(axes).strip('[]').replace(' ', '')

if __name__ == '__main__':
    pass