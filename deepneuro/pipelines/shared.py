import os
import sys
import argparse

from deepneuro.data.data_collection import DataCollection
from deepneuro.models.model import load_old_model
from deepneuro.load.load import load


def DeepNeuroCLI(object):

    def __init__(self):

        self.command_name = 'deepneuro_module'

        self.load()

    def load(self):

        parser = argparse.ArgumentParser(
            description='A number of pre-packaged commands used by the Quantiative Tumor Imaging Lab at the Martinos Center',
            usage='''{} <command> [<args>]

                    The following commands are available:
                       pipeline               Run the entire model pipeline, with options to leave certain pre-processing steps out.
                       docker_pipeline        Run the previous command via a Docker container via nvidia-docker.

                       |Not Implemented|
                       server                 Creates a DeepNeuro server that can process DeepNeuro jobs remotely.
                       explorer               Creates a graphical user interface for this DeepNeuro module.
                '''.format(self.command_name))

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Sorry, that\'s not one of the commands.')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()


def load_data(inputs, output_folder, input_directory=None, ground_truth=None, input_data=None, verbose=True):

    """ In the future, this will need to be modified for multiple types of inputs (i.e. data groups).
    """

    if any(data is None for data in inputs):
        raise ValueError("Cannot run pipeline; required inputs are missing. Please consult this module's documentation, and make sure all required parameters are input.")

    inputs = [os.path.abspath(input_filename) for input_filename in inputs]
    output_folder = os.path.abspath(output_folder)

    input_data = {'input_data': inputs}

    if ground_truth is not None:
        input_data['ground_truth'] = [ground_truth]

    if input_directory is None:

        if any(data is None for data in input_data):
            raise ValueError("Cannot run pipeline; required inputs are missing. Please consult this module's documentation, and make sure all required parameters are input.")

        data_collection = DataCollection(verbose=verbose)
        data_collection.add_case(input_data, case_name=output_folder)

    else:
        data_collection = DataCollection(input_directory, data_group_dict=input_data, verbose=verbose)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if verbose:
        print('File loading completed.')

    return data_collection


def load_model_with_output(model_path=None, model_name=None, outputs=None, postprocessors=None, **kwargs):

    if model_path is not None:
        model = load_old_model(model_path, **kwargs)

    elif model_name is not None:
        model = load_old_model(load(model_name), **kwargs)

    else:
        print('Error. No model provided.')
        return
    
    for output in outputs:
        model.append_output([output])

        for postprocessor in postprocessors:
            output.append_postprocessor([postprocessor]) 

    return model