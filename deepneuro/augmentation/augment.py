import numpy as np

class AugmentationGroup(object):

    # Add reset method.

    def __init__(self, augmentation_dict=None, multiplier=None, total=None, output_shape=None):

        self.iteration = 0

        self.multiplier = multiplier
        self.total = total
        self.output_shape = output_shape

        # For now...
        self.total_iterations = multiplier

        if augmentation_dict is None:
            self.augmentation_dict = []
        else:
            self.augmentation_dict = augmentation_dict
            for key in augmentation_dict:
                augmentation_dict[key].multiplier = multiplier
                augmentation_dict[key].total = total


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

    def augment(self):

        return None

    def initialize_augmentation(self):

        if not self.initialization:
            self.initialization = True

    def iterate(self):

        if self.multiplier is None:
            return

        self.iteration += 1
        if self.iteration == self.multiplier:
            self.iteration = 0

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group

class Copy(Augmentation):

    def __init__(self, patch_shape, patch_extraction_conditions):

        # Get rid of these with super??
        self.multiplier = None
        self.total = None
        self.output_shape = patch_shape

        self.data_groups = {}

# class Shuffle_Values(Augmentation):

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

class Flip_Rotate_2D(Augmentation):

    def __init__(self, data_groups=None, multiplier=None, total=None, flip=True, rotate=True):

        # Get rid of these with super??
        self.multiplier = multiplier
        self.total = total
        self.flip = flip
        self.rotate = rotate

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
        print self.available_transforms

    def initialize_augmentation(self):

        if not self.initialization:

            self.initialization = True

    def iterate(self):

        super(Flip_Rotate_2D, self).iterate()

    def augment(self):

        for data_group in self.data_groups:
            pass

        if self.available_transforms[self.iteration % self.total_transforms, 0]:
            input_data = np.flip(input_data, 2)

        if self.available_transforms[self.iteration % self.total_transforms, 1]:
            input_data = np.rot90(input_data, self.available_transforms[self.iteration % self.total_transforms, 1])

# class ArbitraryRotate3D(Augmentation):

# class SplitDimension(Augmentation):

class ExtractPatches(Augmentation):

    def __init__(self, patch_shape, patch_extraction_conditions):

        # Get rid of these with super??
        self.multiplier = None
        self.total = None
        self.output_shape = patch_shape

        self.initialization = False
        self.iteration = 0

        self.data_groups = {}

        self.patch_shape = patch_shape
        self.patch_extraction_conditions = patch_extraction_conditions
        self.patch_corner = None

    def initialize_augmentation(self):

        if not self.initialization:

            self.condition_list = [-1] * (self.multiplier)
            if self.patch_extraction_conditions is not None:
                start_idx = 0
                for condition_idx, patch_extraction_condition in enumerate(self.patch_extraction_conditions):
                    end_idx = start_idx + int(np.ceil(patch_extraction_condition[1]*self.multiplier))
                    self.condition_list[start_idx:end_idx] = [condition_idx]*(end_idx-start_idx)
                    start_idx = end_idx

            self.initialization = True

    def iterate(self):

        super(ExtractPatches, self).iterate()

        if self.iteration != 0:
            self.generate_patch_corner()
        else:
            self.patch_corner = None

    def augment(self, input_data):

        # Any more sensible way to deal with this case?
        if self.patch_corner == None:
            self.generate_patch_corner()

        # This is repetitive. How to pre-allocate this data?
        patch_slice = [slice(None)] + [slice(None)] + [slice(corner_dim, corner_dim+self.patch_shape[idx], 1) for idx, corner_dim in enumerate(self.patch_corner)]
        output_data = input_data[patch_slice]
        pad_dims = [(0,0), (0,0)]
        for idx, dim in enumerate(output_data.shape[2:]):
            pad_dims += [(0, self.patch_shape[idx]-dim)]

        output_data = np.lib.pad(output_data, tuple(pad_dims), 'edge')

        return output_data

    def generate_patch_corner(self):

        """ Think about how one could to this, say, with 3D and 4D volumes at the same time.
            Also, patching across the modality dimension..? Interesting..
        """

        # TODO: Escape clause in case acceptable patches cannot be found.

        acceptable_patch = False

        data_group_labels = self.data_groups.keys()

        # Hmm... How to make corner search process dimension-agnostic.
        leader_data_group = self.data_groups[data_group_labels[0]]
        data_shape = leader_data_group.current_case.shape

        while not acceptable_patch:

            corner = [np.random.randint(0, max_dim) for max_dim in data_shape[2:]]
            patch_slice = [slice(None)] + [slice(None)] + [slice(corner_dim, corner_dim+self.patch_shape[idx], 1) for idx, corner_dim in enumerate(corner)]

            patches = {}

            # Pad edge patches.
            for key in self.data_groups:
                patches[key] = self.data_groups[key].current_case[patch_slice]
                patch = patches[key] 

                pad_dims = [(0,0), (0,0)]
                for idx, dim in enumerate(patch.shape[2:]):
                    pad_dims += [(0, self.patch_shape[idx]-dim)]

                patch = np.lib.pad(patch, tuple(pad_dims), 'edge')

            if self.condition_list[self.iteration] is not None and self.condition_list[self.iteration] > -1:
                acceptable_patch = self.patch_extraction_conditions[self.condition_list[self.iteration]][0](patches)
            else:
                acceptable_patch = True

        print self.patch_extraction_conditions[self.condition_list[self.iteration]][0]

        self.patch_corner = corner

        return

if __name__ == '__main__':
    pass