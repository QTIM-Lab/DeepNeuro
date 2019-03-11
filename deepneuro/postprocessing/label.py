
import numpy as np

from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes

from deepneuro.postprocessing.postprocessor import Postprocessor
from deepneuro.utilities.util import add_parameter


class BinarizeLabel(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'Binarization')
        add_parameter(self, kwargs, 'postprocessor_string', 'binarized')

        add_parameter(self, kwargs, 'binarization_threshold', 0.5)

    def postprocess(self, input_data, raw_data=None, casename=None):

        return (input_data > self.binarization_threshold).astype(float)


class Rescale(Postprocessor):

    """ TODO: Merge with Normalization Preprocessor
    """

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'Rescale')
        add_parameter(self, kwargs, 'postprocessor_string', 'rescaled')

        add_parameter(self, kwargs, 'input_intensity_range', None)
        add_parameter(self, kwargs, 'output_intensity_range', [0, 255])

    def postprocess(self, input_data, raw_data=None, casename=None):

        normalize_numpy = input_data.astype(float)
            
        if self.input_intensity_range is None:
            input_intensity_range = [np.min(normalize_numpy), np.max(normalize_numpy)]
        else:
            input_intensity_range = self.input_intensity_range

        if input_intensity_range[0] == input_intensity_range[1]:
            normalize_numpy[:] = self.output_intensity_range[0]
            print('Warning: normalization edge case. All array values are equal. Normalizing to minimum value.')
        else:
            normalize_numpy = ((self.output_intensity_range[1] - self.output_intensity_range[0]) * (normalize_numpy - input_intensity_range[0])) / (input_intensity_range[1] - input_intensity_range[0]) + self.output_intensity_range[0] 
        
        if self.input_intensity_range is not None:
            normalize_numpy[normalize_numpy < self.output_intensity_range[0]] = self.output_intensity_range[0]
            normalize_numpy[normalize_numpy > self.output_intensity_range[1]] = self.output_intensity_range[1]

        return normalize_numpy
       

class LargestComponents(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'Binarization')
        add_parameter(self, kwargs, 'postprocessor_string', 'largest_components')

        add_parameter(self, kwargs, 'component_number', 1)
        add_parameter(self, kwargs, 'connectivity', 2)

    def postprocess(self, input_data, raw_data=None, casename=None):

        """ I rewrote Ken's script, but I think I made it worse... TODO: Rewrite again? For clarity.
        """

        for batch in range(input_data.shape[0]):
            for channel in range(input_data.shape[-1]):
                input_data[batch, ..., channel] = largest_components(input_data[batch, ..., channel], component_number=1, connectivity=self.connectivity)

        return input_data


def largest_components(input_data, component_number=1, connectivity=2):

    connected_components = label(input_data, connectivity=connectivity)
    total_components = np.max(connected_components)

    component_sizes = []
    for i in range(1, total_components):
        component_sizes += [np.sum(connected_components == i)]

    component_rankings = np.argsort(np.array(component_sizes))
    component_rankings = component_rankings[:-component_number]

    # I think this would be slower than doing it with one fell swoop,
    # Perhaps with many or statements. Consider doing that.
    for i in component_rankings:
        input_data[connected_components == i + 1] = 0

    return input_data


class FillHoles(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'FillHoles')
        add_parameter(self, kwargs, 'postprocessor_string', 'holes_filled')

        # Hole-Filling Parameters
        add_parameter(self, kwargs, 'slice_dimension', -2)  # Currently not operational

    def postprocess(self, input_data, raw_data=None, casename=None):

        """ Although I don't know, this seems a bit ineffecient. See if there's a better 3D hole-filler out there
            Or better yet, arbitrary dimension hole_filler.
        """

        for batch in range(input_data.shape[0]):
            for channel in range(input_data.shape[-1]):
                for slice_idx in range(input_data.shape[self.slice_dimension]):
                    input_data[batch, ..., slice_idx, channel] = binary_fill_holes(input_data[batch, ..., slice_idx, channel]).astype(np.float)

        return input_data
