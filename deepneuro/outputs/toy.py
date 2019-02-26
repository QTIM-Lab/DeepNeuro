import numpy as np

from deepneuro.utilities.util import add_parameter
from deepneuro.outputs.segmentation import PatchesInference


class PatchDiagram(PatchesInference):

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

        super(PatchDiagram, self).load(kwargs)

        add_parameter(self, kwargs, 'border_width', 1)

    def aggregate_predictions(self, output_data, repatched_image, rep_idx):

        output_data = np.logical_or(output_data, repatched_image).astype(float)
        return output_data

        # output_data += repatched_image * (rep_idx + 1)
        # output_data[output_data == (rep_idx + rep_idx + 1)] = (rep_idx + 1)
        # return output_data

    def run_inference(self, input_patches):

        output_patches = np.ones_like(input_patches)

        front_border_slice = [slice(None)] + [slice(self.border_width, -self.border_width, None) for dim in self.patch_dimensions] + [slice(None)]

        output_patches[tuple(front_border_slice)] = 0

        return output_patches
