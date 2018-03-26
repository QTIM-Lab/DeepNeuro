
import numpy as np

from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes

from deepneuro.postprocessing.postprocessor import Postprocessor
from deepneuro.utilities.util import add_parameter

class BinarizeLabel(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'Binarization')
        add_parameter(self, kwargs, 'postprocessor_string', '_binarized')

        add_parameter(self, kwargs, 'binarization_threshold', 0.5)

    def postprocess(self, input_data):

        return (input_data > self.binarization_threshold).astype(float)

class LargestComponents(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'Binarization')
        add_parameter(self, kwargs, 'postprocessor_string', '_largest_components')

        add_parameter(self, kwargs, 'component_number', 1)
        add_parameter(self, kwargs, 'connectivity', 2)

    def postprocess(self, input_data):

        """ I rewrote Ken's script, but I think I made it worse... TODO: Rewrite again? For clarity.
        """

        for batch in xrange(input_data.shape[0]):
            for channel in xrange(input_data.shape[-1]):

                connected_components = label(input_data[batch, ..., channel], connectivity=self.connectivity)
                total_components = np.max(connected_components)

                component_sizes = []
                for i in xrange(1, total_components):
                    component_sizes += [np.sum(connected_components == i)]

                component_rankings = np.argsort(np.array(component_sizes))
                component_rankings = component_rankings[:-self.component_number]

                # I think this would be slower than doing it with one fell swoop,
                # Perhapse with many or statements. Consider doing that.
                for i in component_rankings:
                    input_data[batch, ..., channel][connected_components == i + 1] = 0

        return input_data

class FillHoles(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'FillHoles')
        add_parameter(self, kwargs, 'postprocessor_string', '_holes_filled')

    def postprocess(self, input_data):

        """ Although I don't know, this seems a bit ineffecient. See if there's a better 3D hole-filler out there
            Or better yet, arbitrary dimension hole_filler.
        """

        for batch in xrange(input_data.shape[0]):
            for channel in xrange(input_data.shape[-1]):

                input_data[batch, ..., channel] = binary_fill_holes(input_data[batch, ..., channel]).astype(np.float)

        return input_data

        # filled_dataa = np.copy(input_data)
        # for i in range(mask.shape[0]):
        #     filled_mask[i,:,:] = binary_fill_holes(mask[i,:,:]).astype(np.float)
        # for j in range(mask.shape[1]):
        #     filled_mask[:,j,:] = binary_fill_holes(mask[:,j,:]).astype(np.float)
        # for k in range(mask.shape[2]):
        #     filled_mask[:,:,k] = binary_fill_holes(mask[:,:,k]).astype(np.float)
        # return filled_mask
   
# def largest_component(mask):
   # conn_comp = label(mask,connectivity=2)
   # unique_val = np.unique(conn_comp)
   # unique_val = unique_val[np.nonzero(unique_val)]
   # count_val = np.zeros(len(unique_val))
   # for i in range(len(unique_val)):
   #     count_val[i] = np.sum(conn_comp == unique_val[i])
   # largest_comp = np.zeros(conn_comp.shape)
   # max_val = unique_val[np.argmax(count_val)]
   # largest_comp[np.where(conn_comp==max_val)] = 1
   # return largest_comp