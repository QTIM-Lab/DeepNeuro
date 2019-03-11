import numpy as np

from random import shuffle
from scipy.sparse import csr_matrix

from deepneuro.utilities.util import add_parameter
from deepneuro.augmentation.augment import Augmentation


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
                shuffle(self.region_list)

            for label, data_group in list(self.data_groups.items()):
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

        if self.patch_region_conditions is not None:
            for region_condition in self.patch_region_conditions:
                self.patch_regions += [np.where(region_condition[0](region_input_data))]
                # self.patch_regions += self.get_indices_sparse(region_condition[0](region_input_data))

        return

    def augment(self, augmentation_num=0):

        # Any more sensible way to deal with this case?
        if self.patches is None:
            self.generate_patch_corner(augmentation_num)

        for label, data_group in list(self.data_groups.items()):

            # A bit lengthy. Also unnecessarily rebuffers patches
            data_group.augmentation_cases[augmentation_num + 1] = self.patches[label]

    def generate_patch_corner(self, augmentation_num=0):

        """ Think about how one could to this, say, with 3D and 4D volumes at the same time.
            Also, patching across the modality dimension..? Interesting..
        """

        # TODO: Escape clause in case acceptable patches cannot be found.

        if self.patch_region_conditions is None:
            corner_idx = None
        else:
            region = self.patch_regions[self.region_list[self.iteration]]
            # print(self.region_list[self.iteration])
            # TODO: Make errors like these more ubiquitous.
            if len(region[0]) == 0:
                # raise ValueError('The region ' + str(self.patch_region_conditions[self.region_list[self.iteration]][0]) + ' has no voxels to select patches from. Please modify your patch-sampling region')
                # Tempfix -- Eek
                region = self.patch_regions[self.region_list[1]]
            if len(region[0]) == 0:
                print('Provided patch extraction region has selected 0 voxels. Selecting non-zero patch.')
                region = np.where(self.data_groups['input_data'].augmentation_cases[augmentation_num] != 0)
                self.patch_regions[self.region_list[0]] = region
            
            corner_idx = np.random.randint(len(region[0]))

        self.patches = {}

        # Pad edge patches.
        for label, data_group in list(self.data_groups.items()):

            input_data = self.data_groups[label].augmentation_cases[augmentation_num]

            # TODO: Some redundancy here
            if corner_idx is None:
                corner = np.array([np.random.randint(0, self.input_shape[label][i]) for i in range(len(self.input_shape[label]))])[self.patch_dimensions[label]]
            else:
                corner = np.array([d[corner_idx] for d in region])[self.patch_dimensions[label]]

            patch_slice = [slice(None)] * (len(self.input_shape[label]) + 1)
            # Will run into problems with odd-shaped patches.
            for idx, patch_dim in enumerate(self.patch_dimensions[label]):
                patch_slice[patch_dim] = slice(max(0, corner[idx] - self.patch_shape[idx] // 2), corner[idx] + self.patch_shape[idx] // 2, 1)

            input_shape = input_data.shape

            self.patches[label] = input_data[tuple(patch_slice)]

            # More complicated padding needed for center-voxel based patches.
            pad_dims = [(0, 0)] * len(self.patches[label].shape)
            for idx, patch_dim in enumerate(self.patch_dimensions[label]):
                pad = [0, 0]
                if corner[idx] > input_shape[patch_dim] - self.patch_shape[idx] // 2:
                    pad[1] = self.patch_shape[idx] // 2 - (input_shape[patch_dim] - corner[idx])
                if corner[idx] < self.patch_shape[idx] // 2:
                    pad[0] = self.patch_shape[idx] // 2 - corner[idx]
                pad_dims[patch_dim] = tuple(pad)

            self.patches[label] = np.lib.pad(self.patches[label], tuple(pad_dims), 'edge')

            # print(self.patches[label].shape)
            # if label == 'ground_truth':
            #     for i in range(4):
            #         print(np.sum(self.patches[label][..., i]))
            # print(label, np.sum(self.patches[label]))

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

            for label, data_group in list(self.data_groups.items()):
                input_shape = data_group.get_shape()
                self.output_shape[label] = np.array(input_shape)
                self.output_shape[label][self.axis[label]] = self.num_chosen
                self.output_shape[label] = tuple(self.output_shape[label])

            self.initialization = True

    def iterate(self):

        super(ChooseData, self).iterate()

    def augment(self, augmentation_num=0):

        choice = None  # This is messed up

        for label, data_group in list(self.data_groups.items()):

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
