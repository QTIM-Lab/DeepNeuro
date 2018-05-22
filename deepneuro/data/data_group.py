import numpy as np

from deepneuro.utilities.conversion import read_image_files


class DataGroup(object):

    def __init__(self, label):

        self.label = label
        self.augmentations = []
        self.data = {}
        self.cases = []
        self.case_num = 0

        # HDF5 variables
        self.source = None
        self.data_casenames = None
        self.data_affines = None

        # TODO: More distinctive naming for "base" and "current" cases.
        self.preprocessed_case = None
        self.preprocessed_affine = None
        self.base_case = None
        self.base_affine = None

        self.augmentation_cases = [None]
        self.augmentation_strings = ['']
        self.preprocessing_data = []

        self.data_storage = None
        self.casename_storage = None
        self.affine_storage = None

        self.output_shape = None
        self.base_shape = None

    def add_case(self, case_name, item):
        self.data[case_name] = item
        self.cases.append(case_name)

    def get_shape(self):

        # TODO: Add support for non-nifti files.
        # Also this is not good. Perhaps specify shape in input?

        if self.output_shape is None:
            if self.data == {}:
                print 'No Data!'
                return (0,)
            elif self.base_shape is None:
                if self.source == 'directory':
                    self.base_shape = read_image_files(self.data.values()[0]).shape
                elif self.source == 'storage':
                    self.base_shape = self.data[0].shape
                self.output_shape = self.base_shape
            else:
                return None
        
        return self.output_shape

    def get_modalities(self):

        if self.data == []:
            return 0
        else:
            return len(self.data[0])

    def get_data(self, index, return_affine=False):

        if self.source == 'directory':
            self.preprocessed_case, affine = read_image_files(self.preprocessed_case, return_affine=True)
            if affine is not None:
                self.preprocessed_affine = affine
            if return_affine:
                return self.preprocessed_case, self.preprocessed_affine
            else:
                return self.preprocessed_case
        elif self.source == 'storage':
            if return_affine:
                return self.data[index][:][np.newaxis], self.data_affines[index]
            else:
                return self.data[index][:][np.newaxis]

        return None

    def get_affine(self, index):

        if self.source == 'directory':
            if self.preprocessed_affine is None:
                self.preprocessed_case, self.preprocessed_affine = read_image_files(self.preprocessed_case, return_affine=True)
            return self.preprocessed_affine
        # A little unsure of the practical implication of the storage code below.
        elif self.source == 'storage':
            return self.data[index][:][np.newaxis], self.data_affines[index]

        return None

    # @profile
    def write_to_storage(self):

        if len(self.augmentation_cases) == 1:
            self.data_storage.append(self.base_case)
        else:
            self.data_storage.append(self.augmentation_cases[-1])

        self.casename_storage.append(np.array(self.base_casename)[np.newaxis][np.newaxis])
        self.affine_storage.append(self.base_affine[:][np.newaxis])
