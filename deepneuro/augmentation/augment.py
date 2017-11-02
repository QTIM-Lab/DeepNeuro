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

            data_group.augmentation_cases[augmentation_num+1] = augmentation_cases[augmentation_num]

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

    def __init__(self, patch_shape, patch_extraction_conditions):

        # Get rid of these with super??
        self.multiplier = None
        self.total = None
        self.output_shape = patch_shape

        self.data_groups = {}

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

    def __init__(self, patch_shape, patch_region_conditions=None, patch_extraction_conditions=None, data_groups=None):

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
        self.patch_regions = []
        self.patches = None
        self.patch_corner = None
        self.patch_slice = None

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

            self.output_shape = {}
            for label, data_group in self.data_groups.iteritems():

                # Might consider doing more flexible patch shapes in the future. E.g. -- different
                # time segments, but the same ground_truth patch.

                # This leading dims business needs to be redone.

                if len(data_group.get_shape()) < 6:
                    self.leading_dims[label] = 0
                    self.output_shape[label] = self.patch_shape + (data_group.get_shape()[-1],)
                else:
                    self.leading_dims[label] = len(data_group.get_shape()) - 5
                    self.output_shape[label] = (data_group.get_shape()[1],) + self.patch_shape + (data_group.get_shape()[-1],)

            self.initialization = True

    def iterate(self):

        super(ExtractPatches, self).iterate()

        self.generate_patch_corner()

    def reset(self, augmentation_num=0):

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
            corner_idx = np.random.randint(len(region[0]))

            # TODO: adjust for non-3D data
            corner = [d[corner_idx] for d in region][:-1]

            self.patches = {}

            # Pad edge patches.
            for key in self.data_groups:

                patch_slice = [slice(corner_dim, corner_dim+self.patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)] + [slice(None)]
                
                self.patches[key] = self.data_groups[key].augmentation_cases[augmentation_num][patch_slice]

                pad_dims = []
                for idx, dim in enumerate(self.patches[key].shape[0:-1]):
                    pad_dims += [(0, self.patch_shape[idx]-dim)]
                pad_dims += [(0,0)]

                self.patches[key]  = np.lib.pad(self.patches[key] , tuple(pad_dims), 'edge')

            print self.patch_region_conditions[self.region_list[self.iteration]][0]

        return

# class Shuffle_Values(Augmentation):

# class ArbitraryRotate3D(Augmentation):

# class SplitDimension(Augmentation):

class GaussianNoise(Augmentation):

    def __init__(self, sigma=.5):

        # Get rid of these with super??
        self.multiplier = None
        self.total = None
        self.output_shape = patch_shape

        self.initialization = False
        self.iteration = 0

        self.total_iterations = multiplier

        self.data_groups = {}

        self.sigma = sigma

    def iterate(self):

        super(Flip_Rotate, self).iterate()

    def augment(self, input_data):

        return input_data

if __name__ == '__main__':
    pass