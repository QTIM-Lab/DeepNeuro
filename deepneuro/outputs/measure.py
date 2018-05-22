import numpy as np

from skimage.morphology import label
from skimage.measure import regionprops, mesh_surface_area, marching_cubes

from deepneuro.utilities.conversion import , read_image_files


def _get_pixdim(pixdim, affine, input_affine, verbose=True):

    """ Currently only functional for 3D images.
    """

    if pixdim is None:
        if affine is not None:
            pixdim = np.abs(affine.diagonal()[0:-1])
        elif input_affine is not None:
            pixdim = np.abs(input_affine.diagonal()[0:-1])
        else:
            if verbose:
                print('Warning -- no resolution provided. Assuming isotropic.')
            return [1, 1, 1]

    return pixdim


def calc_max_2D_diameter(input_data, pixdim=None, affine=None, mask_value=0, axis=2, calc_multiple=False):

    input_data, input_affine = read_image_files(input_data, return_affine=True)

    pixdim = _get_pixdim(pixdim, affine, input_affine)

    input_data = input_data[..., -1]

    connected_components = label(input_data, connectivity=2)
    component_labels = np.unique(connected_components)

    max_2ds = []

    major_diameter = None
    for label_idx in component_labels:

        component = connected_components.astype(int)
        component[connected_components != label_idx] = 0

        major_diameter = None

        for z_slice in xrange(component.shape[2]):

            label_slice = component[..., z_slice]

            if np.sum(label_slice) == 0:
                continue

            label_properties = regionprops(label_slice)
            current_major = label_properties[0].major_axis_length
            current_orientation = label_properties[0].orientation

            x_dim = abs(np.cos(current_orientation) * current_major)
            y_dim = abs(np.sin(current_orientation) * current_major)
            current_major = ((x_dim * pixdim[0])**2 + (y_dim * pixdim[1])**2)**.5

            if major_diameter is None:
                major_diameter = current_major
            elif current_major > major_diameter:
                major_diameter = current_major

        if major_diameter is not None:
            max_2ds += [major_diameter]

    if calc_multiple:
        return max_2ds
    else:
        return max(max_2ds)


def calc_max_3D_diameter(input_data, pixdim=None, affine=None, mask_value=0, axis=2, calc_multiple=False):

    """ Combine repeated code with 2D max diameter?
    """

    input_data, input_affine = read_image_files(input_data, return_affine=True)

    pixdim = _get_pixdim(pixdim, affine, input_affine)

    input_data = input_data[..., -1]

    connected_components = label(input_data, connectivity=2)
    component_labels = np.unique(connected_components)

    max_3ds = []

    major_diameter = None
    for label_idx in component_labels:

        component = connected_components.astype(int)
        component[connected_components != label_idx] = 0

        major_diameter = [None]

        if np.sum(component) == 0:
            continue

        label_properties = regionprops(component)
        current_major = label_properties[0].major_axis_length
        current_orientation = label_properties[0].orientation
        print current_orientation

        if major_diameter is None:
            major_diameter = current_major
        elif current_major > major_diameter:
            major_diameter = current_major

        if major_diameter is not None:
            max_3ds += [major_diameter * pixdim[0] * pixdim[1]]

    if calc_multiple:
        return max_3ds
    else:
        return max(max_3ds)


def calc_surface_area(input_data, pixdim=None, affine=None, mask_value=0, mode='edges'):

    """ Reminder: Verify on real-world data.
        Also, some of the binarization feels clumsy/ineffecient.
        Also, this will over-estimate surface area, because
        it is counting cubes instead of, say, triangular
        surfaces
    """

    input_data, input_affine = read_image_files(input_data, return_affine=True)

    input_data = input_data[..., -1]

    pixdim = _get_pixdim(pixdim, affine, input_affine)
    
    if mode == 'mesh':
        verts, faces = marching_cubes(input_data, 0, pixdim)
        surface_area = mesh_surface_area(verts, faces)

    elif mode == 'edges':
        edges_kernel = np.zeros((3, 3, 3), dtype=float)
        edges_kernel[1, 1, 0] = -1 * pixdim[0] * pixdim[1]
        edges_kernel[0, 1, 1] = -1 * pixdim[1] * pixdim[2]
        edges_kernel[1, 0, 1] = -1 * pixdim[0] * pixdim[2]
        edges_kernel[1, 2, 1] = -1 * pixdim[0] * pixdim[2]
        edges_kernel[2, 1, 1] = -1 * pixdim[1] * pixdim[2]
        edges_kernel[1, 1, 2] = -1 * pixdim[0] * pixdim[1]
        edges_kernel[1, 1, 1] = 1 * (2 * pixdim[0] * pixdim[1] + 2 * pixdim[0] * pixdim[2] + 2 * pixdim[1] * pixdim[2])

        label_numpy = np.copy(input_data)
        label_numpy[label_numpy != mask_value] = 1
        label_numpy[label_numpy == mask_value] = 0

        edge_image = signal.convolve(label_numpy, edges_kernel, mode='same')
        edge_image[edge_image < 0] = 0

        surface_area = np.sum(edge_image)

    else:
        print 'Warning, mode parameter', mode, 'not available. Returning None.'
        surface_area = None

    return surface_area


def calc_voxel_count(input_data, mask_value=0):
    
    input_data = read_image_files(input_data)

    return input_data[input_data != mask_value].size


def calc_volume(input_data, pixdim=None, affine=None, mask_value=0):

    input_data, input_affine = read_image_files(input_data, return_affine=True)

    pixdim = _get_pixdim(pixdim, affine, input_affine)

    return pixdim[0] * pixdim[1] * pixdim[2] * calc_voxel_count(input_data, mask_value)


if __name__ == '__main__':

    pass

