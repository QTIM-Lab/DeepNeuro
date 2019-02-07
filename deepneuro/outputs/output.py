""" Outputs in DeepNeuro are objects that are appended to DeepNeuroModels.
    They will process data passed in by these models, and save their output
    to a provided file location. They may also store metadata about data they
    process longitduinally, to create summary statistics.
"""

import os
import numpy as np

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
        add_parameter(self, kwargs, 'output_directory', None)
        add_parameter(self, kwargs, 'output_filename_base', '_inference.nii.gz')
        add_parameter(self, kwargs, 'stack_outputs', False)

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

        # Derived Parameters
        self.return_objects = []
        self.return_filenames = []
        self.postprocessors = []
        self.postprocessor_string = ''
        self.lead_key = self.inputs[0]
        self.case_directory_output = False
        self.output_extension = nifti_splitext(self.output_filename_base)[1]
        self.open_files = {}
        if self.output_directory is None:
            self.case_directory_output = True

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

        # Very ambiguous naming scheme here.
        # The conditionals are a little cagey here.
        # Also the duplicated code.

        self.generate_output_attributes()

        if self.case is None:

            # At present, this only works for one input, one output patch networks.
            data_generator = self.data_collection.data_generator(verbose=True)

            input_data = next(data_generator)
            
            while input_data is not None:
                self.return_objects = []
                self.return_filenames = []
                self.process_case(input_data[self.lead_key])
                self.postprocess(input_data)
                input_data = next(data_generator)

        else:
            self.return_objects = []
            self.return_filenames = []
            input_data = self.data_collection.get_data(self.case)[self.lead_key]
            self.process_case(input_data)
            self.postprocess(input_data)         

        self.close_output()

        return_dict = {'data': self.return_objects, 'filenames': self.return_filenames}
        return return_dict

    def generate_output_attributes(self):
        
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """

        self.casename = self.data_collection.data_groups[self.lead_key].base_casename
        self.input_affine = self.data_collection.data_groups[self.lead_key].preprocessed_affine
        self.augmentation_string = self.data_collection.data_groups[self.lead_key].augmentation_strings[-1]

        # Create output directory. If not provided, output into original patient folder.
        if self.case_directory_output:
            self.output_directory = os.path.abspath(self.casename)
        else:
            self.output_directory = os.path.abspath(self.output_directory)

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if self.output_extension in ['csv']:
            self.output_filename = self.output_filename_base
        else:
            # Determine output filename based on casename.
            if os.path.exists(self.casename) and not os.path.isdir(self.casename):
                self.output_filename = os.path.basename(nifti_splitext(os.path.abspath(self.casename))[0]) + self.output_filename_base
            elif os.path.isdir(self.casename):
                if self.casename.endswith(os.sep):
                    casename = self.casename[0:-1]
                self.output_filename = os.path.basename(casename) + self.output_filename_base
            else:
                self.output_filename = casename + self.output_filename_base
            
            self.output_filename = os.path.join(self.output_directory, self.output_filename)

        return

    def postprocess(self, input_data):
        """
        
        Parameters
        ----------
        input_data : TYPE
            Description
        """
        if self.save_to_file and (self.save_initial or self.postprocessors == []):
            self.save_output()

        for p_idx, postprocessor in enumerate(self.postprocessors):
            postprocessor.execute(self, raw_data=input_data)
            if self.save_all_steps and p_idx != len(self.postprocessors) - 1:
                self.save_output(p_idx)

        if self.save_to_file and self.postprocessors != []:
            self.save_output(len(self.postprocessors) - 1)

    def open_csv(self, filepath):

        open_file = None

        return open_file

    def close_output(self):

        for key, open_file in self.open_files:
            open_file.close()

        self.open_files = []

    def save_output(self, postprocessor_idx=None):

        # Currently assumes Nifti output. TODO: Make automatically detect output or determine with a class variable.
        # Ideally, split also this out into a saving.py function in utils.
        # Case naming is a little wild here, TODO: make more simple.
        # Some bad practice with abspath here. TODO: abspath all files on input

        for input_data in self.return_objects:

            if self.output_extension in ['csv']:
                if os.path.exists(save_filepath) and not self.replace_existing:
                    continue                
            else:
                # Code for concatenating postprocessor and augmentation strings. Could be condensed.
                postprocessor_string = self.postprocessor_string
                if postprocessor_idx is not None:
                    string_idx = postprocessor_idx
                    while string_idx >= 0:
                        if self.postprocessors[string_idx].postprocessor_string is not None:
                            postprocessor_string += self.postprocessors[string_idx].postprocessor_string
                        string_idx -= 1
                save_filepath = replace_suffix(self.output_filename, '', self.augmentation_string + postprocessor_string)

                # If prediction already exists, skip it. Useful if process is interrupted.
                if os.path.exists(save_filepath) and not self.replace_existing:
                    continue

            if self.show_output:
                # This function call will need to be updated as Outputs is extended for more data types.
                check_data({'prediction': input_data}, batch_size=1, **self.kwargs)

            # Squeeze data here covers for user errors, but could result in unintended outcomes.
            output_shape = input_data.shape
            input_data = np.squeeze(input_data)

            return_filenames = []

            # If there is only one channel, only save one file. Otherwise, attempt to stack outputs or save
            # separate files.
            if output_shape[-1] == 1 or self.stack_outputs:
                self.return_filenames += [save_data(input_data, save_filepath, reference_data=self.input_affine)]
            else:
                for channel in range(output_shape[-1]):
                    return_filenames += [save_data(input_data[..., channel], replace_suffix(save_filepath, input_suffix='', output_suffix='_channel_' + str(channel)), reference_data=self.input_affine)]
                self.return_filenames += [return_filenames]

        return

    def save_to_file(self, input_data, output_filepath):

        # Squeeze data here covers for user errors, but could result in unintended outcomes.
        output_shape = input_data.shape
        input_data = np.squeeze(input_data)

        return_filenames = []

        # If there is only one channel, only save one file. Otherwise, attempt to stack outputs or save
        # separate files.
        if output_shape[-1] == 1 or self.stack_outputs:
            self.return_filenames += [save_data(input_data, output_filepath, reference_data=self.input_affine)]
        else:
            for channel in range(output_shape[-1]):
                return_filenames += [save_data(input_data[..., channel], replace_suffix(output_filepath, input_suffix='', output_suffix='_channel_' + str(channel)), reference_data=self.input_affine)]
            self.return_filenames += [return_filenames]

        return

