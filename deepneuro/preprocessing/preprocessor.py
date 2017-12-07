import os
import numpy as np

from collections import defaultdict

from deepneuro.utilities.util import add_parameter, replace_suffix
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti

class Preprocessor(object):


    def __init__(self, data_groups=None, channel_dim=-1, save_output=True, **kwargs):

        self.output_shape = None
        self.initialization = False

        self.data_groups = {data_group: None for data_group in data_groups}

        self.preprocessor_string = '_convert'

        self.save_output = save_output
        self.channel_dim = channel_dim

        self.outputs = defaultdict(list)

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, case):

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                output_filename = replace_suffix(file, '', self.preprocessor_string)

                array, affine = read_image_files([file], return_affine=True)
                save_numpy_2_nifti(np.squeeze(array), affine, output_filename)

                if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                    os.remove(data_group.preprocessed_case[index])

                data_group.preprocessed_case[index] = output_filename

    def initialize(self):

        if not self.initialization:
            self.initialization = True

    def reset(self):

        self.outputs = defaultdict(list)

        return

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group