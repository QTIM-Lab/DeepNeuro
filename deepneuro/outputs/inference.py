""" This is a base class for running inference on models.
"""

import numpy as np

from deepneuro.outputs.output import Output
from deepneuro.utilities.util import add_parameter


class ModelInference(Output):

    def load(self, kwargs):

        # Evaluation Params
        add_parameter(self, kwargs, 'ground_truth', None)

        # Saving Params
        add_parameter(self, kwargs, 'postprocessor_string', '_inference')

        # Model Parameters
        add_parameter(self, kwargs, 'input_channels', None)
        add_parameter(self, kwargs, 'output_channels', None)

        add_parameter(self, kwargs, 'channels_dim', None)

        if self.channels_dim is None:
            if self.channels_first:
                self.channels_dim = 1
            else:
                self.channels_dim = -1

    def process_case(self, input_data, model=None):
        
        """Summary
        
        Parameters
        ----------
        input_data : TYPE
            Description
        model : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        
        input_data = input_data[self.lead_key]

        if model is not None:
            self.model = model

        if self.channels_first:
            input_data = np.swapaxes(input_data, 1, -1)

        if self.input_channels is not None:
            input_data = np.take(input_data, self.input_channels, self.channels_dim)

        self.output_shape = [1] + list(self.model.model_output_shape)[1:]  # Weird

        output_data = self.predict(input_data)

        if self.output_channels is not None:
            output_data = np.take(output_data, self.output_channels, self.channels_dim)

        # Will fail for time-data.
        if self.channels_first:
            output_data = np.swapaxes(output_data, 1, -1)

        self.return_objects.append(output_data)

        return output_data

    def predict(self, input_data):
        """Summary
        
        Parameters
        ----------
        input_data : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        # Vanilla prediction case is obivously not fleshed out.

        prediction = self.model.predict(input_data)

        return prediction