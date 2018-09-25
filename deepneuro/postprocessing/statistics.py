import csv
import numpy as np

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
        add_parameter(self, kwargs, 'write_mode', 'wb')
        add_parameter(self, kwargs, 'print_output', True)

        self.cost_function_dict = {
            'dice': dice_cost_function
        }

        self.cost_function_label_dict = {
            'dice': 'Dice Coeffecient'
        }

        self.csv_file = open(self.output_log, self.write_mode)
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['casename'] + [self.cost_function_label_dict[cost_function] for cost_function in self.cost_functions])

        # Not sure of the best method to close this file

    def postprocess(self, input_data, raw_data=None, casename=None):

        ground_truth = raw_data[self.ground_truth] 

        if casename is None:
            casename = ''
        output_row = [casename]

        for cost_function in self.cost_functions:

            if cost_function not in self.cost_function_dict.keys():
                print('Error, cost function', cost_function, 'not implemented')

            cost = self.cost_function_dict[cost_function](input_data, ground_truth)

            if self.print_output:
                print(self.cost_function_label_dict[cost_function] + ':', cost)

            output_row += [str(cost)]

        self.csv_writer.writerow(output_row)

        return input_data


def dice_cost_function(input_data, ground_truth):

    im1 = np.asarray(input_data).astype(np.bool)
    im2 = np.asarray(ground_truth).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum