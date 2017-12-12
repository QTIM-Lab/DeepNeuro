import argparse
import sys
import os

from deepneuro.docker.docker_cli import nvidia_docker_wrapper

class Segment_GBM_cli(object):

    def __init__(self):

        parser = argparse.ArgumentParser(
            description='A number of pre-packaged command used by the Quantiative Tumor Imaging Lab at the Martinos Center',
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
            description='''segment pipeline <T2> <T1pre> <T1post> <FLAIR> <output_folder> [-gpu_num <int> -niftis -nobias -preprocessed -keep_outputs]

            Segment an image from DICOMs with all preprocessing steps included.
            -gpu_num <int>      Which CUDA GPU ID # to use.
            -niftis             Input nifti files instead of DIOCM folders.
            -nobias             Skip the bias correction step.
            -preprocessed       Skip bias correction, resampling, and registration.
            -no_ss              [not yet implemented]
            -keep_outputs       Do not delete files generated from intermediary steps.
                ''')

        parser.add_argument('-output_folder', type=str)
        parser.add_argument('-T2', type=str)
        parser.add_argument('-T1', type=str)
        parser.add_argument('-T1POST', type=str)
        parser.add_argument('-FLAIR', type=str)
        parser.add_argument('-input_directory', type=str)
        parser.add_argument('-gpu_num', nargs='?', const='0', type=str)
        parser.add_argument('-bias', action='store_true')  
        parser.add_argument('-resampled', action='store_true')
        parser.add_argument('-registered', action='store_true')
        parser.add_argument('-skullstripped', action='store_true') # Currently non-functional
        parser.add_argument('-normalized', action='store_true') 
        parser.add_argument('-save_preprocess', action='store_true')
        parser.add_argument('-save_all_steps', action='store_true')
        args = parser.parse_args(sys.argv[2:])

        return args
       

    def pipeline(self):

        # segment_gbm pipeline /mnt/jk489/sharedfolder/Duke/Patients/20090501 -T2 /mnt/jk489/sharedfolder/Duke/Patients/20090501/T2 -T1 /mnt/jk489/sharedfolder/Duke/Patients/20090501/T1 -T1POST /mnt/jk489/sharedfolder/Duke/Patients/20090501/T1post -FLAIR /mnt/jk489/sharedfolder/Duke/Patients/20090501/FLAIR -gpu_num 0

        args = self.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

        from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

        predict_GBM(args.output_folder, args.T2, args.T1, args.T1POST, args.FLAIR, None, args.input_directory, bias_corrected=args.bias, resampled=args.resampled, registered=args.registered, skullstripped=args.skullstripped, normalized=args.normalized, save_preprocess=args.save_preprocess, save_all_steps=args.save_all_steps)

    def docker_pipeline(self):

        args = self.parse_args()

        nvidia_docker_wrapper(['segment_gbm', 'pipeline'], vars(args), ['output_folder', 'T2', 'T1', 'T1POST', 'FLAIR', 'input_directory'], docker_container='qtimlab/deepneuro_segment_gbm')

def main():
    Segment_GBM_cli()