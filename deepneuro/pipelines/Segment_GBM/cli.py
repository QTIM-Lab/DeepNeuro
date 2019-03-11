import argparse
import sys
import os

from deepneuro.pipelines.shared import DeepNeuroCLI


class Segment_GBM_cli(DeepNeuroCLI):

    def load(self):

        self.command_name = 'segment_gbm'
        self.docker_container = 'qtimlab/deepneuro_segment_gbm:latest'
        self.filepath_arguments = ['output_folder', 'T1', 'T1POST', 'FLAIR', 'input_directory']

        super(Segment_GBM_cli, self).load()

    def parse_args(self):

        parser = argparse.ArgumentParser(
            description='''segment_gbm pipeline

            Segment an image from DICOMs with all preprocessing steps included.

            -output_folder: A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz"
            -T2, -T1, -T1POST, -FLAIR: Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.
            -wholetumor_output, -enhancing_output: Name of output for wholetumor and enhancing labels, respectively. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "enhancing"
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
        parser.add_argument('-wholetumor_output', nargs='?', type=str, const='wholetumor.nii.gz', default='wholetumor.nii.gz')
        parser.add_argument('-enhancing_output', nargs='?', type=str, const='enhancing.nii.gz', default='enhancing.nii.gz')

        parser.add_argument('-debiased', action='store_true')
        parser.add_argument('-registered', action='store_true')
        parser.add_argument('-skullstripped', action='store_true') 
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

        from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

        predict_GBM(args.output_folder, 
            FLAIR=args.FLAIR, 
            T1POST=args.T1POST, 
            T1PRE=args.T1, 
            ground_truth=None,
            bias_corrected=args.debiased, 
            registered=args.registered, 
            skullstripped=args.skullstripped, 
            preprocessed=args.preprocessed, 
            save_only_segmentations=args.save_only_segmentations, 
            output_probabilities=args.output_probabilities,
            save_all_steps=args.save_all_steps, 
            output_wholetumor_filename=args.wholetumor_output, 
            output_enhancing_filename=args.enhancing_output, 
            quiet=args.quiet)


def main():
    Segment_GBM_cli()