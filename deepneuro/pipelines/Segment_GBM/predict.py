import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model

from deepneuro.load.load import load
from deepneuro.data.data_collection import DataCollection
from deepneuro.preprocessing.signal import N4BiasCorrection
from deepneuro.preprocessing.transform import Resample, Coregister

# Temporary
from keras.utils import plot_model
import glob

def predict_GBM(output_folder, T2=None, T1=None, T1POST=None, FLAIR=None, ground_truth=None, input_directory=None, dicoms=True, bias=True, resampled=False, registered=False, skullstripped=False, normalized=False, save_steps=False):

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

    preprocessing_steps = []

    if bias:
        preprocessing_steps += [N4BiasCorrection(data_groups=['input_modalities'])]

    if not resampled:
        preprocessing_steps += [Resample(data_groups=['input_modalities'])]

    if not registered:
        preprocessing_steps += [Coregister(data_groups=['input_modalities'], reference_channel = 1)]

    data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 3, Segmentation
    #--------------------------------------------------------------------#

    wholetumor_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, 'wholetumor_segmentation.nii.gz'),
                        'batch_size': 50,
                        'patch_overlaps': 1,
                        'channels_first': True,
                        'patch_dimensions': [-3,-2,-1],
                        'input_channels': [0, 3]}

    enhancing_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, 'enhancing_segmentation.nii.gz'),
                        'batch_size': 50,
                        'patch_overlaps': 1,
                        'channels_first': True,
                        'patch_dimensions': [-3,-2,-1]}

    wholetumor_model = load_old_model('/mnt/jk489/sharedfolder/segmentation/qtim_ChallengePipeline/model_files/wholetumor_FLAIRT1post.h5')
    enhancing_model = load_old_model('enhancingtumor_BRATS_submission.h5')
    # wholetumor_model = load_old_model(load('Segment_GBM_wholetumor'))
    # enhancing_model = load_old_model(load('Segment_GBM_enhancing'))

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
    T2, T1, T1POST, FLAIR = [os.path.join(output_folder, file) for file in ['T2_raw.nii.gz', 'T1_raw.nii.gz', 'T1post_raw.nii.gz', 'FLAIR_raw.nii.gz']]

    predict_GBM(output_folder, T2, T1, T1POST, FLAIR)