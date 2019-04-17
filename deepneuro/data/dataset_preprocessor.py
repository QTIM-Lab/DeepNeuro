import os
import numpy as np

from deepneuro.utilities.util import add_parameter, replace_suffix, cli_sanitize, docker_print
from deepneuro.utilities.conversion import read_image_files, save_data


class DatasetPreprocessor(object):

    """ 
    """

    def __init__(self, **kwargs):

        # File-Saving Parameters
        add_parameter(self, kwargs, 'overwrite', True)
        add_parameter(self, kwargs, 'save_output', False)
        add_parameter(self, kwargs, 'output_folder', None)
        
        # Input Parameters
        add_parameter(self, kwargs, 'file_input', False)

        # Naming Parameters
        add_parameter(self, kwargs, 'name', 'Conversion')
        add_parameter(self, kwargs, 'preprocessor_string', '_convert')

        # Internal Parameters
        add_parameter(self, kwargs, 'data_groups', None)
        add_parameter(self, kwargs, 'verbose', False)

        # Derived Parameters
        self.array_input = True

        self.output_data = None
        self.output_affines = None
        self.output_shape = None
        self.output_filenames = []
        self.initialization = False

        # Dreams of linked lists here.
        self.data_dictionary = None
        self.next_prepreprocessor = None
        self.previous_preprocessor = None
        self.order_index = 0

        self.load(kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return
