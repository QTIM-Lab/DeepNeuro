import csv
import numpy as np

from skimage.measure import label

from deepneuro.postprocessing.postprocessor import Postprocessor
from deepneuro.utilities.util import add_parameter


class ErrorCalculation(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'ErrorCalculation')
        add_parameter(self, kwargs, 'postprocessor_string', '')

        # Logging Parameters
        add_parameter(self, kwargs, 'output_log', 'outputs.csv')
        add_parameter(self, kwargs, 'cost_functions', ['dice'])
        add_parameter(self, kwargs, 'write_mode', 'w')
        add_parameter(self, kwargs, 'print_output', True)

        self.cost_function_dict = {
            'dice': dice_cost_function,
            'accuracy': accuracy_cost_function,
            'cluster_accuracy': cluster_accuracy_cost_function
        }

        self.cost_function_label_dict = {
            'dice': 'Dice Coeffecient',
            'accuracy': 'Accuracy',
            'cluster_accuracy': 'Cluster Accuracy'
        }

        self.transform_output = False
        self.csv_file = None
        # Not sure of the best method to close this file

    def postprocess(self, input_data, raw_data=None, casename=None):

        if self.csv_file is None:
            self.csv_file = open(self.output_log, self.write_mode)
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['casename'] + [self.cost_function_label_dict[cost_function] for cost_function in self.cost_functions])

        ground_truth = raw_data[self.ground_truth] 

        if casename is None:
            casename = ''
        output_row = [casename]

        for cost_function in self.cost_functions:

            if cost_function not in list(self.cost_function_dict.keys()):
                print(('Error, cost function', cost_function, 'not implemented'))

            cost = self.cost_function_dict[cost_function](input_data, ground_truth)

            if self.print_output:
                print((self.cost_function_label_dict[cost_function] + ':', cost))

            output_row += [str(cost)]

        self.csv_writer.writerow(output_row)
        self.csv_file.flush()

        return input_data

    def close(self):
        self.csv_file.close()


class RanoCalculation(Postprocessor):

    def load(self, kwargs):

        # Naming parameter
        add_parameter(self, kwargs, 'name', 'RanoCalculation')
        add_parameter(self, kwargs, 'postprocessor_string', '_RANO')

        # Logging Parameters
        add_parameter(self, kwargs, 'output_log', 'rano.csv')
        add_parameter(self, kwargs, 'write_mode', 'w')
        add_parameter(self, kwargs, 'print_output', True)

        self.transform_output = False
        self.csv_file = None
        # Not sure of the best method to close this file

    def postprocess(self, input_data, raw_data=None, casename=None):

        if self.csv_file is None:
            self.csv_file = open(self.output_log, self.write_mode)
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['casename'] + [self.cost_function_label_dict[cost_function] for cost_function in self.cost_functions])

        if casename is None:
            casename = ''
        output_row = [casename]

        input_data = raw_data[self.inputs]

        for cost_function in self.cost_functions:

            if cost_function not in list(self.cost_function_dict.keys()):
                print(('Error, cost function', cost_function, 'not implemented'))

            cost = self.cost_function_dict[cost_function](input_data, ground_truth)

            if self.print_output:
                print((self.cost_function_label_dict[cost_function] + ':', cost))

            output_row += [str(cost)]

        self.csv_writer.writerow(output_row)
        self.csv_file.flush()

        return input_data

    def close(self):
        self.csv_file.close()    


# def rano_calculation(input_label):

#     # Should this class be defined in-function?
#     class Point(namedtuple('Point', 'x y')):
#         __slots__ = ()
#         @property
#         def length(self):
#             return (self.x ** 2 + self.y ** 2) ** 0.5
        
#         def __sub__(self, p):
#             return Point(self.x - p.x, self.y - p.y)
        
#         def __str__(self):
#             return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)

#     def vector_norm(p):
#         length = p.length
#         return Point(p.x / length, p.y / length)

#     def compute_pairwise_distances(P1, P2, min_length=10):
        
#         euc_dist_matrix = cdist(P1, P2, metric='euclidean')
#         indices = []
#         for x in range(euc_dist_matrix.shape[0]):
#             for y in range(euc_dist_matrix.shape[1]):

#                 p1 = Point(*P1[x])
#                 p2 = Point(*P1[y])
#                 d = euc_dist_matrix[x, y]
                
#                 if p1 == p2 or d < min_length:
#                     continue

#                 indices.append([p1, p2, d])

#         return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)

# import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# from PIL import Image
# from skimage.measure import find_contours
# from skimage.morphology import binary_erosion, disk
# from scipy.spatial.distance import cdist
# import pandas as pd
# from collections import namedtuple


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


# def rano(binary_image, tol=0.01, output_file=None, background_image=None):

#     # Estimate lesion height and width
#     h, w = np.sum(np.max(binary_image > 0, axis=1)), np.sum(np.max(binary_image > 0, axis=0))

#     # Dilate slightly to prevent self-intersections, and compute contours
#     dilated = binary_erosion(binary_image, disk(radius=1)).astype('uint8') * 255
#     contours = find_contours(dilated, level=1)

#     if len(contours) == 0:
#         print "No lesion contours > 1 pixel detected."
#         return 0.0

#     # Calculate pairwise distances over boundary
#     outer_contour = np.round(contours[0]).astype(int)  # this assumption should always hold...
#     euc_dist_matrix, ordered_diameters = compute_pairwise_distances(outer_contour, outer_contour, min_length=w)

#     # Exhaustive search for longest valid line segment and its orthogonal counterpart
#     try:
#         p1, p2, q1, q2 = find_largest_orthogonal_cross_section(ordered_diameters, binary_image, tolerance=tol)
#         rano_measure = ((p2 - p1).length * (q2 - q1).length)
#     except TypeError:
#         print "Error: unable to compute RANO measurement"
#         return 0.0

#     if output_file is not None:

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

#     return rano_measure


# if __name__ == '__main__':

#     import sys
#     try:
#         img = np.asarray(Image.open(sys.argv[1]))
#         if len(img.shape) == 3:
#             img = img[:,:,0]
#         print("Computing RANO...")
#         r = rano(img, output_file='rano.png')
#         print('RANO measurement: {:.2f}'.format(r))
#     except IndexError as e:
#         print("Please specify a valid image file!")
#         print(e)


def dice_cost_function(input_data, ground_truth):
    
    """ Calculate the dice coefficient.
    
    Parameters
    ----------
    input_data, ground_truth : NumPy
        Arrays to be compared.
    
    Returns
    -------
    float
        Dice coefficient.
    
    """
    
    im1 = np.asarray(input_data).astype(np.bool)
    im2 = np.asarray(ground_truth).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: Predicted data and ground truth must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def accuracy_cost_function(input_data, ground_truth):

    """Summary
    
    Parameters
    ----------
    input_data, ground_truth : NumPy
        Arrays to be compared.
    
    Returns
    -------
    TYPE
        Description
    """

    return np.sum(input_data == ground_truth)


def cluster_accuracy_cost_function(input_data, ground_truth, connectivity=2):
    
    """Computes a function to see how many clusters that exist in the ground truth
        data have overlapping segments in the input data. Note that this does not account
        for extraneous segmentations in the input data that do not correspond to any
        clusters in the ground truth data.
    
    Parameters
    ----------
    input_data, ground_truth : NumPy
        Arrays to be compared.
    connectivity : int, optional
        Description
    
    Returns
    -------
    float
        Cluster accuracy metric.
    """

    if input_data.shape[-1] != 1:
        raise NotImplementedError('Cluster accuracy not implemented for data with multiple channels.')

    input_data = input_data[0, ..., 0]
    ground_truth = ground_truth[0, ..., 0]

    overlapping_map = np.logical_and(input_data, ground_truth)
    connected_components = label(ground_truth, connectivity=connectivity)
    total_components = np.max(connected_components)

    overlapping_components = 0

    for i in range(1, total_components + 1):
        individual_component = np.copy(connected_components)
        individual_component[individual_component != i] == 0
        if np.sum(np.logical_and(overlapping_map, individual_component.astype(bool))) != 0:
            overlapping_components += 1

    return overlapping_components / total_components
