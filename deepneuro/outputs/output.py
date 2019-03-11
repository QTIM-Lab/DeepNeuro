""" Outputs in DeepNeuro are objects that are appended to DeepNeuroModels.
    They will process data passed in by these models, and save their output
    to a provided file location. They may also store metadata about data they
    process longitduinally, to create summary statistics.
"""

import os
import csv
import numpy as np

from collections import defaultdict

from deepneuro.utilities.conversion import save_data
from deepneuro.utilities.util import add_parameter, replace_suffix, nifti_splitext, additional_kwargs
from deepneuro.utilities.visualize import check_data


class Output(object):

    def __init__(self, **kwargs):
        
        """Output object.
        
        Parameters
        ----------
        data_collection: DataCollection
            DataCollection object that Output is currently predicting on. Default
            is None
        inputs: str, list
            Key or list of keys that designate input data in DataCollections. Default
            is ['input_data']
        ground_truth: str, list
            Key or list of keys that designate ground truth in DataCollections. Default
            is ['ground_truth']
        save_to_file: bool
            Save output to disc. Default is True.
        save_initial: bool
            Save unpostprocessed output to disc, if postprocessors are applied.
        save_all_steps: bool
            Save all intermediate postprocessing steps. Filenames are designated
            by postprocessor strings.
        output_directory: str
            Directory to output inference in to. If None, outputs to directories
            defined by the 'casename' attribute in DataCollections, if present.
        output_filename_base: str
            Predictions will use this filename for output, potentially with suffixes
            attached in postprocessors or augmentations are specified.
        stack_outputs: bool
            If True, DeepNeuro will attempt to combined multiple channels into one
            output file where possible.
        case_in_filename : bool

        channels_first: bool

        """
        
        # Data Parameters
        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'inputs', ['input_data'])
        add_parameter(self, kwargs, 'ground_truth', ['ground_truth'])

        # Saving Parameters
        add_parameter(self, kwargs, 'save_to_file', True)
        add_parameter(self, kwargs, 'save_initial', False)
        add_parameter(self, kwargs, 'save_all_steps', False)
        add_parameter(self, kwargs, 'output_directory', './')
        add_parameter(self, kwargs, 'output_filename', None)
        add_parameter(self, kwargs, 'output_filename_base', 'prediction.png')
        add_parameter(self, kwargs, 'stack_outputs', False)
        add_parameter(self, kwargs, 'case_in_filename', True)

        # Visualization Parameters
        add_parameter(self, kwargs, 'show_output', False)
        add_parameter(self, kwargs, 'show_output_save', False)

        # Implementation Parameters
        add_parameter(self, kwargs, 'channels_first', False)
        add_parameter(self, kwargs, 'batch_size', 32)

        # Internal Parameters
        add_parameter(self, kwargs, 'replace_existing', True)
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'case', None)

        if self.output_filename is not None:
            self.output_filename_base = self.output_filename

        # Derived Parameters
        self.return_objects = []
        self.return_filenames = []
        self.postprocessors = []
        self.postprocessor_string = ''
        self.lead_key = self.inputs[0]
        self.case_directory_output = False
        self.output_filename_base, self.output_extension = nifti_splitext(self.output_filename_base)
        self.open_files = defaultdict(lambda: None)

        self.load(kwargs)

        self.kwargs = additional_kwargs(self, kwargs)

        return

    def load(self, kwargs):

        """ This method is used by children classes to load additional attributes from kwargs. These
            may be parameters specific to a certain model type, for example.
        """

        return

    def execute(self, model):

        return None

    def append_postprocessor(self, postprocessors):

        if type(postprocessors) is not list:
            postprocessors = [postprocessors]

        for postprocessor in postprocessors:
            self.postprocessors += [postprocessor]

        return None

    def generate(self):

        self.generate_output_directory()

        if self.case is not None:
            return self.generate_individual_case(self.case)

        data_generator = self.data_collection.data_generator(verbose=True, batch_size=self.batch_size)

        input_data = next(data_generator)
        
        while input_data is not None:
            self.return_objects = []
            self.return_filenames = []
            self.process_case(input_data)
            self.postprocess(input_data)
            input_data = next(data_generator)       

        self.close_output()

        return_dict = {'data': self.return_objects, 'filenames': self.return_filenames}
        return return_dict

    def generate_individual_case(self, case):

        self.return_objects = []
        self.return_filenames = []
        input_data = self.data_collection.get_data(self.case)
        self.process_case(input_data)
        self.postprocess(input_data)         

        self.close_output()

        return_dict = {'data': self.return_objects, 'filenames': self.return_filenames}
        return return_dict

    def generate_output_directory(self):

        self.output_directory = os.path.abspath(self.output_directory)

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        return

    def generate_filenames(self, data, postprocessor_idx=None):

        output_filenames = []
        for idx in range(len(data['casename'])):

            casename = data['casename'][idx]
            augmentation_string = data[self.lead_key + '_augmentation_string'][idx]

            if self.case_in_filename:
                fileparts = [casename, self.output_filename_base]
            else:
                fileparts = [self.output_filename_base]

            if augmentation_string != '':
                fileparts = fileparts + [augmentation_string]
            if postprocessor_idx is not None:
                if [self.postprocessors[postprocessor_idx].postprocessor_string] != '':
                    fileparts = fileparts + [self.postprocessors[postprocessor_idx].postprocessor_string]

            output_filename = '_'.join(fileparts)

            if self.output_extension not in ['.csv']:
                output_filename = os.path.join(self.output_directory, output_filename + self.output_extension)

            output_filenames += [output_filename]

        return output_filenames

    def postprocess(self, input_data):
        """
        
        Parameters
        ----------
        input_data : TYPE
            Description
        """
        if self.save_to_file and (self.save_initial or self.postprocessors == []):
            self.save_output(raw_data=input_data)

        for p_idx, postprocessor in enumerate(self.postprocessors):
            postprocessor.execute(self, raw_data=input_data)
            if self.save_all_steps and p_idx != len(self.postprocessors) - 1 and postprocessor.transform_output:
                self.save_output(p_idx, raw_data=input_data)

        if self.save_to_file and self.postprocessors != []:
            self.save_output(len(self.postprocessors) - 1, raw_data=input_data)

    def open_csv(self, filepath):

        open_file = None

        return open_file

    def close_output(self):

        for key, open_file in self.open_files:
            open_file.close()

        self.open_files = defaultdict(lambda: None)

    def save_output(self, postprocessor_idx=None, raw_data=None):

        # Currently assumes Nifti output. TODO: Make automatically detect output or determine with a class variable.
        # Ideally, split also this out into a saving.py function in utils.

        for input_data in self.return_objects:

            output_filenames = self.generate_filenames(raw_data, postprocessor_idx)

            if self.output_extension in ['.csv']:
                self.save_to_csv(input_data, output_filenames)  
            else:
                self.save_to_disk(input_data, output_filenames, raw_data)

            if self.show_output:
                # This function call will need to be updated as Outputs is extended for more data types.
                check_data({'prediction': input_data}, batch_size=1, **self.kwargs)

        return

    def save_to_csv(self, input_data, output_filenames):

        if os.path.exists(self.output_filename_base) and not self.replace_existing:
            return              

        if self.open_files[self.lead_key] is None:
            self.open_files[self.lead_key] = open(self.output_filename_base + self.output_extension, 'w')

        writer = csv.writer(self.open_files[self.lead_key])

        for row_idx, row in enumerate(input_data):
            output_row = [output_filenames[row_idx]] + list(row)
            writer.writerow(output_row)

        return

    def save_to_disk(self, input_data, output_filenames, raw_data):

        for batch_idx, batch in enumerate(input_data):

            # Squeeze data here covers for user errors, but could result in unintended outcomes.
            output_shape = batch.shape
            batch = np.squeeze(batch)
            input_affine = raw_data[self.lead_key + '_affine'][batch_idx]

            output_filename = output_filenames[batch_idx]
            return_filenames = []

            # If there is only one channel, only save one file. Otherwise, attempt to stack outputs or save
            # separate files.
            if output_shape[-1] == 1 or (output_shape[-1] == 3 and batch.ndim == 3) or self.stack_outputs:
                self.return_filenames += [save_data(batch, output_filename, reference_data=input_affine)]
            else:
                for channel in range(output_shape[-1]):
                    return_filenames += [save_data(batch[..., channel], replace_suffix(output_filename, input_suffix='', output_suffix='_channel_' + str(channel)), reference_data=input_affine)]
                self.return_filenames += [return_filenames]

        return

