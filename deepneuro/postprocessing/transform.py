import numpy as np

from deepneuro.postprocessing.postprocessor import Postprocessor
from deepneuro.utilities.util import add_parameter


class UniqueClasses(Postprocessor):

    """This class reverses the effect of one-hot encoding data. 
    """
    
    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'UniqueClasses')
        add_parameter(self, kwargs, 'postprocessor_string', '_unique_classes')

    def postprocess(self, input_data, raw_data=None, casename=None):

        output_data = np.argmax(input_data, axis=1)

        return output_data