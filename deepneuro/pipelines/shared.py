"""
"""

import os
import sys
import argparse

from shutil import copy

from deepneuro.data.data_collection import DataCollection
from deepneuro.container.container_cli import nvidia_docker_wrapper


class DeepNeuroCLI(object):

    def __init__(self):

        self.command_name = 'deepneuro_module'
        self.docker_container = 'qtimlab/deepneuro:latest'
        self.filepath_arguments = []

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

    def docker_pipeline(self):

        args = self.parse_args()

        nvidia_docker_wrapper([self.command_name, 'pipeline'], vars(args), self.filepath_arguments, docker_container=self.docker_container)


def load_data(inputs, output_folder, input_directory=None, ground_truth=None, input_data=None, verbose=True):

    """ A convenience function when building single-input pipelines. This function
        quickly builds DataCollections
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


def create_Dockerfile(output_directory, models_included=None, module_name=None, deepneuro_branch='master'):

    current_dir = os.path.realpath(os.path.dirname(__file__))
    base_Dockerfile = os.path.join(current_dir, 'Dockerfile_base')
    new_Dockerfile = os.path.join(output_directory, 'Dockerfile')

    copy(base_Dockerfile, new_Dockerfile)

    echo_count = os.path.join(current_dir, 'echo_count.txt')
    with open(echo_count, 'r') as myfile:
        echo_count = myfile.read()

    with open(new_Dockerfile, "a") as writefile:

        if models_included is not None:

            if module_name is None:
                raise ValueError("If you are including models in your container, please include the module_name parameter.")

            writefile.write("RUN mkdir -p /home/DeepNeuro/deepneuro/load/{}\n".format(module_name))

            for key, value in models_included.items():
                writefile.write("""RUN wget -O /home/DeepNeuro/deepneuro/load/{}/{}.h5 {}""".format(module_name, key, value))

        writefile.write("""
        RUN echo {} \n
        RUN git pull \n
        RUN python3 /home/DeepNeuro/setup.py develop \n
\n
        # Commands at startup. \n
        WORKDIR "/" \n
        RUN chmod 777 /home/DeepNeuro/entrypoint.sh \n
        ENTRYPOINT ["/home/DeepNeuro/entrypoint.sh"]""".format(echo_count))

    return new_Dockerfile


def create_Singularity(output_directory, docker_name):

    output_singularity = os.path.join(output_directory, 'Singularity.' + docker_name)

    with open("output_singularity", "w") as writefile:
        writefile.write("Bootstrap: docker\n")
        writefile.write("From: qtimlab/{}:latest".format(docker_name))

    return output_singularity


def upload_icon(output_directory, icon_filepath):

    resources_directory = os.path.join(output_directory, 'resources')

    if not os.path.exists(resources_directory):
        os.mkdir(os.path.join(output_directory, 'resources'))

    copy(icon_filepath, os.path.join(resources_directory, 'icon.png'))