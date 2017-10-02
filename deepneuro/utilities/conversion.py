
import numpy as np
import nibabel as nib

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy




def read_image_files(image_files, return_affine=False):

    # Rename this function to something more descriptive?

    image_list = []
    affine = None
    for image_file in image_files:
        image_list.append(convert_input_2_numpy(image_file))
        # if affine

    # This is a little clunky.
    if return_affine:
        # This assumes all images share an affine matrix.
        # Replace with a better convert function, at some point.
        return np.stack([image for image in image_list], axis=-1), nib.load(image_files[0]).affine
    else:
        return np.stack([image for image in image_list], axis=-1)