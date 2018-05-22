
import os
import urllib

current_dir = os.path.realpath(os.path.dirname(__file__))

# Perhaps one day replace this with config files distributed at the level of modules.
dataset_dict = {'Segment_GBM_downsampled_wholetumor': [os.path.join(current_dir, 'Segment_GBM', 'downsampled_wholetumor.h5'), "https://www.dropbox.com/s/h2pj9hzruz3hq9a/wholetumor_downsampled_BRATS_submission.h5?dl=1"],
                'Segment_GBM_upsample_wholetumor': [os.path.join(current_dir, 'Segment_GBM', 'upsample_wholetumor.h5'), "https://www.dropbox.com/s/r4pelwcsscnuvc9/upsample_wholetumor_BRATS_submission.h5?dl=1"],
                'Segment_GBM_wholetumor': [os.path.join(current_dir, 'Segment_GBM', 'wholetumor.h5'), "https://www.dropbox.com/s/74tjx14ue11rc0q/wholetumor.h5?dl=1"],
                'Segment_GBM_enhancing': [os.path.join(current_dir, 'Segment_GBM', 'enhancing.h5'), "https://www.dropbox.com/s/usdal6cbkw3bceu/enhancingtumor_BRATS_submission.h5?dl=1"],
                'Skull_Strip_T1Post_FLAIR': [os.path.join(current_dir, 'Skull_Strip', 'Skull_Stripping.h5'), "https://www.dropbox.com/s/gn2w4u4vw1orcyj/FLAIRT1post_ss.h5?dl=1"],
                'gbm_wholetumor_mri': [os.path.join(current_dir, 'Segment_GBM', 'Segment_GBM_Wholetumor_Model.h5'), "https://www.dropbox.com/s/bnbdi1yogq2yye3/GBM_Wholetumor_Public.h5?dl=1"],
                'gbm_enhancingtumor_mri': [os.path.join(current_dir, 'Segment_GBM', 'Segment_GBM_Enhancing_Model.h5'), "https://www.dropbox.com/s/hgsqi0vj7cfuk1g/GBM_Enhancing_Public.h5?dl=1"],
                }


def load(dataset):

    """Summary
    
    Parameters
    ----------
    dataset : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    # TODO: Loading progress indicator.

    dataset_path = dataset_dict[dataset][0]

    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    if not os.path.exists(dataset_path):
        try:
            urllib.urlretrieve(dataset_dict[dataset][1], dataset_path)
        except Exception:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            raise Exception

    return dataset_path

# class Segment_GBM_cli(object):

#     def __init__(self):

#         parser = argparse.ArgumentParser(
#             description='Load pre-trained models and datasets provided by DeepNeuro.',
#             usage='''load 

#                     The following commands are available:
#                        pipeline               Run the entire segmentation pipeline, with options to leave certain pre-processing steps out.
#                        docker_pipeline        Run the previous command via a Docker container via nvidia-docker.
#                 ''')

#         parser.add_argument('command', help='Subcommand to run')
#         args = parser.parse_args(sys.argv[1:2])

#         if not hasattr(self, args.command):
#             print 'Sorry, that\'s not one of the commands.'
#             parser.print_help()
#             exit(1)

#         # use dispatch pattern to invoke method with same name
#         getattr(self, args.command)()

#     def parse_args(self):

#         parser = argparse.ArgumentParser(
#             description='''segment_gbm pipeline

#             Segment an image from DICOMs with all preprocessing steps included.

#             -output_folder: A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz"
#             -T2, -T1, -T1POST, -FLAIR: Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.
#             -wholetumor_output, -enhancing_output: Name of output for wholetumor and enhancing labels, respectively. Should not be a filepath, like '/home/user/enhancing.nii.gz', but just a name, like "enhancing"
#             -gpu_num: Which CUDA GPU ID # to use. Defaults to 0, i.e. the first gpu.
#             -debiased: If flagged, data is assumed to already have been N4 bias-corrected, and skips that preprocessing step.
#             -resampled: If flagged, data is assumed to already have been isotropically resampled, and skips that preprocessing step.
#             -registered: If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step.
#             -save_all_steps: If flagged, intermediate volumes in between preprocessing steps will be saved in output_folder.
#             -save_preprocessed: If flagged, the final volume after all preprocessing steps will be saved in output_folder
#                 ''')

#         parser.add_argument('-output_folder', type=str)
#         parser.add_argument('-output_probabilities', action='store_true')
#         args = parser.parse_args(sys.argv[2:])

#         return args

#     def pipeline(self):

#         args = self.parse_args()

#         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

#         from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

#         predict_GBM(args.output_folder, args.T1, args.T1POST, args.FLAIR, None, args.input_directory, bias_corrected=args.debiased, resampled=args.resampled, registered=args.registered, skullstripped=args.skullstripped, preprocessed=args.normalized, save_preprocess=args.save_preprocess, save_all_steps=args.save_all_steps, output_wholetumor_filename=args.wholetumor_output, output_enhancing_filename=args.enhancing_output)

#     def docker_pipeline(self):

#         args = self.parse_args()

#         nvidia_docker_wrapper(['segment_gbm', 'pipeline'], vars(args), ['output_folder', 'T1', 'T1POST', 'FLAIR', 'input_directory'], docker_container='qtimlab/deepneuro_segment_gbm:latest')


# def main():
#     Segment_GBM_cli()

if __name__ == '__main__':
    pass