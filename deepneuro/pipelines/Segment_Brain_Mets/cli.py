""" Command line interface for the Brain Metastases Segmentation
    module.
"""

import argparse
import sys
import os

from deepneuro.docker.docker_cli import nvidia_docker_wrapper


class Segment_Mets_cli(object):

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='A number of pre-packaged commands used by the Quantiative Tumor Imaging Lab at the Martinos Center',
            usage='''segment_gbm <command> [<args>]

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

        return

    def parse_args(self):

        parser = argparse.ArgumentParser(
            description='''segment_gbm pipeline

            Segment an image from DICOMs with all preprocessing steps included.

            -output_folder: A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz"
            -T1, -T1POST, -FLAIR: Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.
            -segmentation_output: Name of output for enhancing tumor segmentations. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "segmentation"
            -gpu_num: Which CUDA GPU ID # to use. Defaults to 0, i.e. the first gpu.
            -debiased: If flagged, data is assumed to already have been N4 bias-corrected, and skips that preprocessing step.
            -resampled: If flagged, data is assumed to already have been isotropically resampled, and skips that preprocessing step.
            -registered: If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step.
            -skullstripped: If flagged, data is assumed to have been already skull-stripped.
            -preprocessed: If flagged, data is assumed to have been preprocessed with with zero mean and unit variance with respect to a brain mask.
            -save_all_steps: If flagged, intermediate volumes in between preprocessing steps will be saved in output_folder.
            -save_preprocessed: If flagged, the final volume after all preprocessing steps will be saved in output_folder
                ''')

        parser.add_argument('-output_folder', type=str)
        parser.add_argument('-T1', type=str)
        parser.add_argument('-T1POST', type=str)
        parser.add_argument('-FLAIR', type=str)
        parser.add_argument('-input_directory', type=str)
        parser.add_argument('-segmentation_output', nargs='?', type=str, const='segmentation.nii.gz', default='segmentation.nii.gz')
        parser.add_argument('-gpu_num', nargs='?', const='0', default='0', type=str)
        parser.add_argument('-debiased', action='store_true')  
        parser.add_argument('-resampled', action='store_true')
        parser.add_argument('-registered', action='store_true')
        parser.add_argument('-skullstripped', action='store_true') 
        parser.add_argument('-preprocessed', action='store_true') 
        parser.add_argument('-save_preprocess', action='store_true')
        parser.add_argument('-save_all_steps', action='store_true')
        parser.add_argument('-output_probabilities', action='store_true')
        args = parser.parse_args(sys.argv[2:])

        return args

    def pipeline(self):

        args = self.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

        from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

        predict_GBM(args.output_folder, T1POST=args.T1POST, T1PRE=args.T1, FLAIR=args.FLAIR, ground_truth=None, input_directory=args.input_directory, bias_corrected=args.debiased, resampled=args.resampled, registered=args.registered, skullstripped=args.skullstripped, preprocessed=args.preprocessed, save_preprocess=args.save_preprocess, save_all_steps=args.save_all_steps, output_segmentation_filename=args.segmentation_output)

    def docker_pipeline(self):

        args = self.parse_args()

        nvidia_docker_wrapper(['segment_mets', 'pipeline'], vars(args), ['output_folder', 'T1', 'T1POST', 'FLAIR', 'input_directory'], docker_container='qtimlab/deepneuro_segment_mets:latest')


def main():
    Segment_Mets_cli()