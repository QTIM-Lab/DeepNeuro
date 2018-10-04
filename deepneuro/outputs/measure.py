import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

from skimage.morphology import label
from skimage.measure import regionprops, mesh_surface_area, marching_cubes
# from skimage.morphology import label, binary_erosion, disk
# from skimage.measure import regionprops, mesh_surface_area, marching_cubes, find_contours
from scipy import signal
# from scipy.spatial.distance import cdist
# from collections import namedtuple

from deepneuro.utilities.conversion import read_image_files


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


# class Point(namedtuple('Point', 'x y')):

#     __slots__ = ()
#     @property
#     def length(self):
#         return (self.x ** 2 + self.y ** 2) ** 0.5
    
#     def __sub__(self, p):
#         return Point(self.x - p.x, self.y - p.y)
    
#     def __str__(self):
#         return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)


# def plot_contours(contours, lw=4, alpha=0.5):
#     for n, contour in enumerate(contours):
#         plt.plot(contour[:, 1], contour[:, 0], linewidth=lw, alpha=alpha)


# def vector_norm(p):
#     length = p.length
#     return Point(p.x / length, p.y / length)


# def compute_pairwise_distances(P1, P2, min_length=10):
    
#     euc_dist_matrix = cdist(P1, P2, metric='euclidean')
#     indices = []
#     for x in range(euc_dist_matrix.shape[0]):
#         for y in range(euc_dist_matrix.shape[1]):

#             p1 = Point(*P1[x])
#             p2 = Point(*P1[y])
#             d = euc_dist_matrix[x, y]
            
#             if p1 == p2 or d < min_length:
#                 continue

#             indices.append([p1, p2, d])

#     return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)


# def interpolate(p1, p2, d):
    
#     X = np.linspace(p1.x, p2.x, round(d)).astype(int)
#     Y = np.linspace(p1.y, p2.y, round(d)).astype(int)
#     XY = np.asarray(list(set(zip(X, Y))))
#     return XY


# def find_largest_orthogonal_cross_section(pairwise_distances, img, tolerance=0.01):

#     for i, (p1, p2, d1) in enumerate(pairwise_distances):

#         # Compute intersections with background pixels
#         XY = interpolate(p1, p2, d1)
#         intersections = np.sum(img[x, y] == 0 for x, y in XY)

#         if intersections == 0:

#             V = vector_norm(Point(p2.x - p1.x, p2.y - p1.y))
            
#             # Iterate over remaining line segments
#             for j, (q1, q2, d2) in enumerate(pairwise_distances[i:]):
                
#                 W = vector_norm(Point(q2.x - q1.x, q2.y - q1.y))
#                 if abs(np.dot(V, W)) < tolerance:
                    
#                     XY = interpolate(q1, q2, d2)
#                     intersections = np.sum(img[x, y] == 0 for x, y in XY)
                    
#                     if intersections == 0:
#                         return p1, p2, q1, q2


# def calc_2D_RANO_measure(input_data, pixdim=None, affine=None, mask_value=0, axis=2, calc_multiple=False, background_image=None, output_filepath=None, verbose=True):

#     """ Finds a RANO measure by an exhaustive search of all boundary poinrts on all lesions and all slices.
#     """

#     input_data, input_affine = read_image_files(input_data, return_affine=True)

#     pixdim = _get_pixdim(pixdim, affine, input_affine)

#     input_data = input_data[..., -1]

#     connected_components = label(input_data, connectivity=2)
#     component_labels = np.unique(connected_components)

#     max_2ds = []
#     max_2d_images = []

#     major_diameter = None
#     for label_idx in component_labels:

#         component = connected_components.astype(int)
#         component[connected_components != label_idx] = 0

#         major_diameter = None

#         for z_slice in xrange(component.shape[2]):

#             label_slice = component[..., z_slice]

#             if np.sum(label_slice) == 0:
#                 continue

#             p1, p2, q1, q2 = calc_rano_points(label_slice, pixdim=pixdim)

#             x_dim = abs(np.cos(current_orientation) * current_major)
#             y_dim = abs(np.sin(current_orientation) * current_major)
#             current_major = ((x_dim * pixdim[0])**2 + (y_dim * pixdim[1])**2)**.5

#             if major_diameter is None:
#                 major_diameter = current_major
#             elif current_major > major_diameter:
#                 major_diameter = current_major

#         if major_diameter is not None:
#             max_2ds += [major_diameter]

#     if output_filepath is not None:

#         fig = plt.figure(figsize=(10, 10), frameon=False)
#         plt.margins(0,0)
#         plt.gca().set_axis_off()
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         if background_image is not None:
#             plt.imshow(background_image, cmap='gray')
#         else:
#             plt.imshow(binary_image, cmap='gray')
#         plot_contours(contours, lw=1, alpha=1.)
#         D1 = np.asarray([[p1.x, p2.x], [p1.y, p2.y]])
#         D2 = np.asarray([[q1.x, q2.x], [q1.y, q2.y]])
#         plt.plot(D1[1, :], D1[0, :], lw=2, c='r')
#         plt.plot(D2[1, :], D2[0, :], lw=2, c='r')
#         plt.text(20, 20, 'RANO: {:.2f}'.format(rano_measure), {'color': 'r', 'fontsize': 20})
#         plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0, dpi=100) 

#     if calc_multiple:
#         return max_2ds
#     else:
#         return max(max_2ds)


# def calc_rano_points(binary_image, tol=0.01, output_file=None, background_image=None, verbose=False):

#     """ Code developed by James Brown, postdoctoral fellow at the QTIM lab.
#     """

#     # Estimate lesion height and width
#     height, width = np.sum(np.max(binary_image > 0, axis=1)), np.sum(np.max(binary_image > 0, axis=0))

#     # Dilate slightly to prevent self-intersections, and compute contours
#     dilated = binary_erosion(binary_image, disk(radius=1)).astype('uint8') * 255
#     contours = find_contours(dilated, level=1)

#     if len(contours) == 0:
#         if verbose:
#             print "No lesion contours > 1 pixel detected."
#         return 0.0

#     # Calculate pairwise distances over boundary
#     outer_contour = np.round(contours[0]).astype(int)  # this assumption should always hold...
#     euc_dist_matrix, ordered_diameters = compute_pairwise_distances(outer_contour, outer_contour, min_length=width)

#     # Exhaustive search for longest valid line segment and its orthogonal counterpart
#     try:
#         p1, p2, q1, q2 = find_largest_orthogonal_cross_section(ordered_diameters, binary_image, tolerance=tol)
#     except TypeError:
#         if verbose:
#             print "Error: unable to compute RANO measurement"
#         return 0.0

#     return p1, p2, q1, q2


def calc_max_2D_diameter_ellipse(input_data, pixdim=None, affine=None, mask_value=0, axis=2, calc_multiple=False):

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

        for z_slice in range(component.shape[2]):

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
        print(current_orientation)

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
        print(('Warning, mode parameter', mode, 'not available. Returning None.'))
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

