from deepneuro.utilities.util import add_parameter


class Postprocessor(object):

    def __init__(self, **kwargs):

        # Default Variables
        add_parameter(self, kwargs, 'verbose', False)
        add_parameter(self, kwargs, 'raw_data', None)
        add_parameter(self, kwargs, 'ground_truth', 'ground_truth')

        # Naming Variables
        add_parameter(self, kwargs, 'name', 'Postprocesser')
        add_parameter(self, kwargs, 'postprocessor_string', 'postprocess')

        self.transform_output = True

        self.load(kwargs)

    def load(self, kwargs):

        return

    def execute(self, output, raw_data):

        postprocessed_objects = []

        # TODO: Return object syntax is broken / not implemented
        for return_object in output.return_objects:

            if self.verbose:
                print(('Postprocessing with...', self.name))

            # This piece of code has not yet been refactored.
            if self.ground_truth in list(raw_data.keys()):
                casename = raw_data['casename'][0]
            else:
                casename = None

            postprocessed_objects += [self.postprocess(return_object, raw_data=raw_data, casename=casename)]

        output.return_objects = postprocessed_objects

    def postprocess(self, input_data, raw_data=None, casename=None):

        return input_data

    def clear_outputs(self):

        return

    def close(self):

        return