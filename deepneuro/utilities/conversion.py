
import numpy as np
import nibabel as nib
import math

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

def round_up(x, y):
    return int(math.ceil(float(size) / float(stride)))

def read_image_files(image_files, return_affine=False):

    # Rename this function to something more descriptive?

    image_list = []
    affine = None
    for image_file in image_files:
        image_list.append(convert_input_2_numpy(image_file))
        # if affine

    if image_list[0].ndim == 4:
        array = np.rollaxis(np.stack([image for image in image_list], axis=-1), 3, 0)
    else:
        array = np.stack([image for image in image_list], axis=-1)

    # This is a little clunky.
    if return_affine:
        affine = nib.load(image_files[0]).affine
        # This assumes all images share an affine matrix.
        # Replace with a better convert function, at some point.
        return array, affine
    else:
        return array