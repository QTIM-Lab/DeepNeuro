import numpy as np

from deepneuro.outputs.inference import ModelInference
from deepneuro.utilities.util import add_parameter


class ClassInference(ModelInference):

    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        super(ClassInference, self).load(kwargs)

        # Patching Parameters
        add_parameter(self, kwargs, 'patch_overlaps', 1)
        add_parameter(self, kwargs, 'input_patch_shape', None)
        add_parameter(self, kwargs, 'output_patch_shape', None)
        add_parameter(self, kwargs, 'check_empty_patch', True)
        add_parameter(self, kwargs, 'pad_borders', True)

        add_parameter(self, kwargs, 'patch_dimensions', None)

        add_parameter(self, kwargs, 'output_patch_dimensions', self.patch_dimensions)

