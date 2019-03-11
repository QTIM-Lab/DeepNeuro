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
