import os
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''segment pipeline <T2> <T1pre> <T1post> <FLAIR> <output_folder> [-gpu_num <int> -niftis -nobias -preprocessed -keep_outputs]

        Segment an image from DICOMs with all preprocessing steps included.

        -output_folder: A filepath to your output folder. Two nifti files will be generated "enhancingtumor.nii.gz" and "wholetumor.nii.gz"
        -T2, -T1, -T1POST, -FLAIR: Filepaths to input MR modalities. Inputs can be either nifti files or DICOM folders. Note that DICOM folders should only contain one volume each.
        -gpu_num: Which CUDA GPU ID # to use. Defaults to 0, i.e. the first gpu.
        -debiased: If flagged, data is assumed to already have been N4 bias-corrected, and skips that preprocessing step.
        -resampled: If flagged, data is assumed to already have been isotropically resampled, and skips that preprocessing step.
        -registered: If flagged, data is assumed to already have been registered into the same space, and skips that preprocessing step.
        -save_all_steps: If flagged, intermediate volumes in between preprocessing steps will be saved in output_folder.
        -save_preprocessed: If flagged, the final volume after all preprocessing steps will be saved in output_folder
            ''')

    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--T2', type=str)
    parser.add_argument('--T1', type=str)
    parser.add_argument('--T1POST', type=str)
    parser.add_argument('--FLAIR', type=str)
    parser.add_argument('--gpu_num', nargs='?', const='0', type=str)
    parser.add_argument('--debiased', action='store_true')  
    parser.add_argument('--resampled', action='store_true')
    parser.add_argument('--registered', action='store_true')
    parser.add_argument('--skullstripped', action='store_true') # Currently non-functional
    parser.add_argument('--normalized', action='store_true') 
    parser.add_argument('--save_preprocess', action='store_true')
    parser.add_argument('--save_all_steps', action='store_true')
    parser.add_argument('--ModelName', type=str)
    parser.add_argument('--Output_WholeTumor', type=str)
    parser.add_argument('--Output_EnhancingTumor', type=str)
    args = parser.parse_args(sys.argv[2:])
    #

    args = self.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

    from deepneuro.pipelines.Segment_GBM.predict import predict_GBM

    predict_GBM(args.output_folder, T2=args.T2, T1=args.T1, T1POST=args.T1POST, FLAIR=args.FLAIR, ground_truth=None, input_directory='/INPUT_DATA', bias_corrected=args.debiased, resampled=args.resampled, registered=args.registered, skullstripped=args.skullstripped, normalized=False, save_preprocess=False, save_all_steps=False, output_wholetumor_filename=args.Output_WholeTumor, output_enhancing_filename=args.Output_EnhancingTumor)