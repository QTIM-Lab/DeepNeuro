import argparse
import sys
import os

from deepneuro.docker.docker_cli import nvidia_docker_wrapper


class Skull_Stripping_cli(object):

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='A number of pre-packaged commands used by the Quantiative Tumor Imaging Lab at the Martinos Center',
            usage='''segment <command> [<args>]

                    The following commands are available:
                       pipeline               Run the entire segmentation pipeline, with options to leave certain pre-processing steps out.
                       docker_pipeline        Run the previous command via a Docker container via nvidia-docker.
                ''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print 'Sorry, that\'s not one of the commands.'
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

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
        parser.add_argument('-mask_output', nargs='?', type=str, const='skullstrip_mask', default='skullstrip_mask.nii.gz')
        parser.add_argument('-gpu_num', nargs='?', const='0', default='0', type=str)
        parser.add_argument('-debiased', action='store_true')  
        parser.add_argument('-resampled', action='store_true')
        parser.add_argument('-registered', action='store_true')
        parser.add_argument('-normalized', action='store_true') 
        parser.add_argument('-save_preprocess', action='store_true')
        parser.add_argument('-save_all_steps', action='store_true')
        parser.add_argument('-output_probabilities', action='store_true')
        args = parser.parse_args(sys.argv[2:])

        return args
       
    def pipeline(self):

        args = self.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

        from deepneuro.pipelines.Skull_Stripping.predict import skull_strip

        skull_strip(args.output_folder, args.T1POST, args.FLAIR, None, args.input_directory, bias_corrected=args.debiased, resampled=args.resampled, registered=args.registered, normalized=args.normalized, save_preprocess=args.save_preprocess, save_all_steps=args.save_all_steps, mask_output=args.mask_output)

    def docker_pipeline(self):

        args = self.parse_args()

        nvidia_docker_wrapper(['skull_strip', 'pipeline'], vars(args), ['output_folder', 'T1POST', 'FLAIR', 'input_directory'], docker_container='qtimlab/deepneuro_skull_strip:latest')


def main():
    Skull_Stripping_cli()