""" A set of postprocessors constructed specifically for classification tasks.
"""

import numpy as np

from deepneuro.postprocessing.postprocessor import Postprocessor
from deepneuro.utilities.util import add_parameter


class MaximumClassifier(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'MaxClass')
        add_parameter(self, kwargs, 'postprocessor_string', 'max_class')

    def postprocess(self, input_data, raw_data=None, casename=None):

        return np.argmax(input_data, axis=1)[..., None]