from deepneuro.utilities.util import add_parameter


class Postprocessor(object):

    def __init__(self, **kwargs):

        # Default Variables
        add_parameter(self, kwargs, 'verbose', False)

        # Naming Variables
        add_parameter(self, kwargs, 'name', 'Postprocesser')
        add_parameter(self, kwargs, 'postprocessor_string', '_postprocess')

        self.load(kwargs)

    def load(self, kwargs):

        return

    def execute(self, output):

        postprocessed_objects = []

        for return_object in output.return_objects:

            if self.verbose:
                print('Postprocessing with...', self.name)

            postprocessed_objects += [self.postprocess(return_object)]

        output.return_objects = postprocessed_objects

    def postprocess(self, input_data):

        return input_data

    def clear_outputs(self):

        return