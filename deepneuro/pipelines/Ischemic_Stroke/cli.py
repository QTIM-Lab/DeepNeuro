""" Command line interface for the Brain Metastases Segmentation
    module.
"""

import argparse
import sys
import os

from deepneuro.pipelines.shared import DeepNeuroCLI


class Segment_Ischemic_Stroke_cli(DeepNeuroCLI):

    def load(self):

        self.command_name = 'segment_ischemic_stroke'
        self.docker_container = 'qtimlab/deepneuro_segment_ischmeic_stroke:latest'
        self.filepath_arguments = ['output_folder', 'DWI', 'B0', 'input_directory']

        super(Segment_Ischemic_Stroke_cli, self).load()

    def parse_args(self):

        parser = argparse.ArgumentParser(
            description='''segment_ischemic_stroke pipeline

            Create a binary ischemic-stroke mask.

            -output_folder: A filepath to your output folder, where segmentations and precussor files will be output.
            -DWI, B0: Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.
            -segmentation_output: Name of output for enhancing tumor segmentations. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "segmentation.nii.gz"
            -gpu_num: Which CUDA GPU ID # to use. Defaults to 0, i.e. the first gpu.
            -registered: If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step.
            -normalized: If flagged, data is assumed to have been preprocessed with with zero mean and unit variance with respect to a brain mask.
            -save_all_steps: If flagged, intermediate volumes in between preprocessing steps will be saved in output_folder.
            -save_preprocessed: If flagged, the final volume after all preprocessing steps will be saved in output_folder
                ''')

        parser.add_argument('-output_folder', type=str)
        parser.add_argument('-DWI', type=str)
        parser.add_argument('-B0', type=str)
        parser.add_argument('-input_directory', type=str)
        parser.add_argument('-segmentation_output', nargs='?', type=str, const='segmentation.nii.gz', default='segmentation.nii.gz')

        parser.add_argument('-registered', action='store_true')
        parser.add_argument('-preprocessed', action='store_true') 
        
        parser.add_argument('-gpu_num', nargs='?', const='0', default='0', type=str)
        parser.add_argument('-save_only_segmentations', action='store_true')
        parser.add_argument('-save_all_steps', action='store_true')
        parser.add_argument('-quiet', action='store_true')
        parser.add_argument('-output_probabilities', action='store_true')
        args = parser.parse_args(sys.argv[2:])

        return args

    def pipeline(self):

        args = self.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

        from deepneuro.pipelines.Ischemic_Stroke.predict import predict_ischemic_stroke

        predict_ischemic_stroke(args.output_folder, DWI=args.DWI, B0=args.B0, ground_truth=None, input_directory=args.input_directory, registered=args.registered, preprocessed=args.preprocessed, save_all_steps=args.save_all_steps, output_segmentation_filename=args.segmentation_output, quiet=args.quiet, save_only_segmentations=args.save_only_segmentations)


def main():
    Segment_Ischemic_Stroke_cli()