import os
import numpy as np

from collections import defaultdict

from deepneuro.utilities.util import add_parameter, replace_suffix
from deepneuro.utilities.conversion import read_image_files, save_numpy_2_nifti

class Preprocessor(object):


    def __init__(self, data_groups=None, channel_dim=-1, save_output=True, overwrite=False, verbose=False, **kwargs):

        self.output_shape = None
        self.initialization = False

        self.data_groups = {data_group: None for data_group in data_groups}

        self.overwrite = overwrite
        self.save_output = save_output
        self.channel_dim = channel_dim

        add_parameter(self, kwargs, 'name', 'Conversion')
        self.verbose = verbose

        add_parameter(self, kwargs, 'preprocessor_string', '_convert')

        self.outputs = defaultdict(list)
        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, case):

        """ There is a lot of repeated code in the preprocessors. Think about preprocessor structures and work on this class.
        """

        self.initialize() # TODO: make overwrite work with initializations

        for label, data_group in self.data_groups.iteritems():

            for index, file in enumerate(data_group.preprocessed_case):

                if self.verbose:
                    print 'Preprocessor: ', self.name, 'Case: ', file
                    sys.stdout.flush()

                self.base_file = file # Weird name for this, make more descriptive
                self.output_filename = replace_suffix(file, '', self.preprocessor_string)

                # if not os.path.exists(self.output_filename) or overwrite:
                continue_status = self.preprocess() # Really weird

                if not continue_status:

                    if not self.save_output and data_group.preprocessed_case[index] != data_group.data[case][index]:
                        os.remove(data_group.preprocessed_case[index])

                    data_group.preprocessed_case[index] = self.output_filename

                    self.outputs['outputs'] += [self.output_filename]

    def preprocess(self):

        array, affine = read_image_files([self.base_file], return_affine=True)
        save_numpy_2_nifti(np.squeeze(array), affine, self.output_filename)

    def initialize(self):

        return

    def reset(self):

        self.outputs = defaultdict(list)

        return

    def append_data_group(self, data_group):
        self.data_groups[data_group.label] = data_group