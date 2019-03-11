import argparse
import sys
import os

from deepneuro.pipelines.shared import DeepNeuroCLI


class Skull_Stripping_cli(DeepNeuroCLI):

    def load(self):

        self.command_name = 'skull_stripping'
        self.docker_container = 'qtimlab/deepneuro_skull_strip:latest'
        self.filepath_arguments = ['output_folder', 'T1POST', 'FLAIR', 'input_directory']

        super(Skull_Stripping_cli, self).load()

    def parse_args(self):

        parser = argparse.ArgumentParser(
            description='''skull_strip pipeline <T1post> <FLAIR> <output_folder> [-gpu_num <int> -niftis -nobias -preprocessed -keep_outputs]

            Segment an image from DICOMs with all preprocessing steps included.

            -output_folder: A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz"
            -T1POST, -FLAIR: Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.
            -mask_output: Name of output for your binary skull mask. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "enhancing"
            -gpu_num: Which CUDA GPU ID # to use. Defaults to 0, i.e. the first gpu.
            -debiased: If flagged, data is assumed to already have been N4 bias-corrected, and skips that preprocessing step.
            -resampled: If flagged, data is assumed to already have been isotropically resampled, and skips that preprocessing step.
            -registered: If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step.
            -save_all_steps: If flagged, intermediate volumes in between preprocessing steps will be saved in output_folder.
            -save_preprocessed: If flagged, the final volume after all preprocessing steps will be saved in output_folder
                ''')

        parser.add_argument('-output_folder', type=str)
        parser.add_argument('-T1POST', type=str)
        parser.add_argument('-FLAIR', type=str)
        parser.add_argument('-input_directory', type=str)
        parser.add_argument('-segmentation_output', nargs='?', type=str, const='segmentation.nii.gz', default='segmentation.nii.gz')

        parser.add_argument('-debiased', action='store_true')
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

        from deepneuro.pipelines.Skull_Stripping.predict import skull_strip

        skull_strip(output_folder=args.output_folder, T1POST=args.T1POST, FLAIR=args.FLAIR, ground_truth=None, input_directory=args.input_directory, bias_corrected=args.debiased, registered=args.registered, preprocessed=args.preprocessed, save_only_segmentations=args.save_only_segmentations, save_all_steps=args.save_all_steps, output_segmentation_filename=args.segmentation_output, quiet=args.quiet)


def main():
    Skull_Stripping_cli()