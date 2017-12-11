import os

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model

from deepneuro.load.load import load
from deepneuro.data.data_collection import DataCollection

from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.preprocessing.signal import N4BiasCorrection, ZeroMeanNormalization
from deepneuro.preprocessing.transform import Resample, Coregister
from deepneuro.preprocessing.skullstrip import SkullStrip

def predict_GBM(output_folder, T2=None, T1=None, T1POST=None, FLAIR=None, ground_truth=None, input_directory=None, bias_corrected=False, resampled=False, registered=False, skullstripped=False, normalized=False, preprocessed=False, save_preprocess=True, save_all_steps=False):

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    input_data = {'input_modalities': [FLAIR, T2, T1, T1POST]}

    if ground_truth is not None:
        input_data['ground_truth'] = [ground_truth]

    if input_directory is None:

        if any(data is None for data in input_data):
            raise ValueError("Cannot segment GBM. Please specify all four modalities.")

        data_collection = DataCollection(verbose=True)
        data_collection.add_case(input_data, case_name=output_folder)

    else:
        data_collection = DataCollection(input_directory, modality_dict=input_data, verbose=True)
        data_collection.fill_data_groups()

    #--------------------------------------------------------------------#
    # Step 2, Preprocess Data
    #--------------------------------------------------------------------#

    save_all_steps = True
    save_preprocess = True

    if not preprocessed:
        print 'ABOUT TO PREPROCESS....'

        # Random hack to save DICOMs to niftis for further processing.
        preprocessing_steps = [Preprocessor(data_groups=['input_modalities'], save_output=save_all_steps)]

        if not bias_corrected:
            preprocessing_steps += [N4BiasCorrection(data_groups=['input_modalities'], save_output=save_all_steps)]

        if not resampled:
            preprocessing_steps += [Resample(data_groups=['input_modalities'], save_output=save_all_steps)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_modalities'], save_output=save_all_steps, reference_channel = 1)]

        if not skullstripped:
            preprocessing_steps += [SkullStrip(data_groups=['input_modalities'], save_output=save_all_steps, reference_channel = 1)]

        if not normalized:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_modalities'], save_output=save_preprocess, mask=preprocessing_steps[-1], preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 3, Segmentation
    #--------------------------------------------------------------------#

    wholetumor_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, 'wholetumor_segmentation.nii.gz'),
                        'batch_size': 75,
                        'patch_overlaps': 8,
                        'channels_first': True,
                        'patch_dimensions': [-3,-2,-1],
                        'output_patch_shape': (1,26,26,26),
                        # 'input_channels': [0, 3],
                        }

    enhancing_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, 'enhancing_segmentation.nii.gz'),
                        'batch_size': 75,
                        'patch_overlaps': 8,
                        'channels_first': True,
                        'output_patch_shape': (1,26,26,26),
                        'patch_dimensions': [-3,-2,-1]}

    wholetumor_model = load_old_model(load('Segment_GBM_wholetumor'))
    enhancing_model = load_old_model(load('Segment_GBM_enhancing'))

    wholetumor_prediction = ModelPatchesInference(data_collection, **wholetumor_prediction_parameters)
    wholetumor_model.append_output([wholetumor_prediction])

    enhancing_prediction = ModelPatchesInference(data_collection, **enhancing_prediction_parameters)
    enhancing_model.append_output([enhancing_prediction])

    for case in data_collection.cases:

        print case

        wholetumor_prediction.case = case
        wholetumor_file = wholetumor_model.generate_outputs()[0][0]

        print '\n'

        data_collection.add_channel(case, wholetumor_file)

        print '\n'

        enhancing_prediction.case = case
        enhancing_file = enhancing_model.generate_outputs()[0]

        print '\nNext Case\n'

if __name__ == '__main__':

    input_directory = '/mnt/jk489/sharedfolder/BRATS2017/Test_Case_2'
    T2, T1, T1POST, FLAIR = '*T2_raw.*', '*T1_raw.*', '*T1post_raw.*', 'FLAIR_raw.*'

    output_folder = '/mnt/jk489/sharedfolder/BRATS2017/Test_Case_3/Brats17_CBICA_AQO_1'
    output_folder = '/mnt/jk489/sharedfolder/Duke/Patients/20090501'
    # T2, T1, T1POST, FLAIR = [os.path.join(output_folder, file) for file in ['T2_raw.nii.gz', 'T1_raw.nii.gz', 'T1post_raw.nii.gz', 'FLAIR_raw.nii.gz']]
    # T2, T1, T1POST, FLAIR = [os.path.join(output_folder, file) for file in ['T2_pp.nii.gz', 'T1_pp.nii.gz', 'T1post_pp.nii.gz', 'FLAIR_pp.nii.gz']]
    T2, T1, T1POST, FLAIR = [os.path.join(output_folder, file) for file in ['T2', 'T1', 'T1post', 'FLAIR']]

    predict_GBM(output_folder, T2, T1, T1POST, FLAIR, input_directory=None)