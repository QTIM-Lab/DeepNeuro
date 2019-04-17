from deepneuro.postprocessing.postprocessor import Postprocessor
from deepneuro.utilities.util import add_parameter


class MaximumSlice(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'MaximumSlice')
        add_parameter(self, kwargs, 'postprocessor_string', '_maxslice')

        # Max parameters
        add_parameter(self, kwargs, '')

    def postprocess(self, input_data, raw_data=None, casename=None):

        volume_data = raw_data[self.inputs]
        input_data_sums = np.

        return (input_data > self.binarization_threshold).astype(float)