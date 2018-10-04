import argparse
import sys
import os

from deepneuro.docker.docker_cli import nvidia_docker_wrapper


class {{cli_class_name}}(object):

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='A number of pre-packaged commands used by the Quantiative Tumor Imaging Lab at the Martinos Center',
            usage='''{{cli_command_name}} <command> [<args>]

                    The following commands are available:
                       pipeline               Run the entire segmentation pipeline, with options to leave certain pre-processing steps out.
                       docker_pipeline        Run the previous command via a Docker container via nvidia-docker.
                ''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Sorry, that\'s not one of the commands.')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def parse_args(self):

        parser = argparse.ArgumentParser(
            description=\
            {{method_description}}
        )

        {{arguments}}

        args = parser.parse_args(sys.argv[2:])

        return args

    def pipeline(self):

        args = self.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

        {{import_command}}

        {{function_call}}

    def docker_pipeline(self):

        args = self.parse_args()

       {{docker_call}}


def main():
    {{cli_class_name}}()