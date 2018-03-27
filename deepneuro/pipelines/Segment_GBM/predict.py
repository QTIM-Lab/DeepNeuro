import os

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model
from deepneuro.load.load import load
from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.preprocessing.signal import N4BiasCorrection, ZeroMeanNormalization
from deepneuro.preprocessing.transform import Resample, Coregister
from deepneuro.preprocessing.skullstrip import SkullStrip
from deepneuro.postprocessing.label import BinarizeLabel, LargestComponents, FillHoles

def predict_GBM(output_folder, T2=None, T1=None, T1POST=None, FLAIR=None, ground_truth=None, input_directory=None, bias_corrected=True, resampled=False, registered=False, skullstripped=False, normalized=False, preprocessed=False, save_preprocess=False, save_all_steps=False, output_wholetumor_filename='wholetumor_segmentation.nii.gz', output_enhancing_filename='enhancing_segmentation.nii.gz', verbose=True):

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    input_data = {'input_modalities': [FLAIR, T2, T1, T1POST]}

    if ground_truth is not None:
        input_data['ground_truth'] = [ground_truth]

    if input_directory is None:

        if any(data is None for data in input_data):
            raise ValueError("Cannot segment GBM. Please specify all four modalities.")

        data_collection = DataCollection(verbose=verbose)
        data_collection.add_case(input_data, case_name=output_folder)

    else:
        data_collection = DataCollection(input_directory, modality_dict=input_data, verbose=verbose)
        data_collection.fill_data_groups()

    #--------------------------------------------------------------------#
    # Step 2, Preprocess Data
    #--------------------------------------------------------------------#

    if not preprocessed:
        print 'ABOUT TO PREPROCESS....'

        # Random hack to save DICOMs to niftis for further processing.
        preprocessing_steps = [Preprocessor(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not bias_corrected:
            preprocessing_steps += [N4BiasCorrection(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not resampled:
            preprocessing_steps += [Resample(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=1)]

        if not skullstripped:
            preprocessing_steps += [SkullStrip(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=1)]

        if not normalized:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, mask=preprocessing_steps[-1], output_folder=output_folder, preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 3, Segmentation
    #--------------------------------------------------------------------#

    wholetumor_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, output_wholetumor_filename),
                        'batch_size': 75,
                        'patch_overlaps': 1,
                        'channels_first': True,
                        'patch_dimensions': [-3,-2,-1],
                        'output_patch_shape': (1,26,26,26),
                        # 'input_channels': [0, 3],
                        }

    enhancing_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, output_enhancing_filename),
                        'batch_size': 75,
                        'patch_overlaps': 1,
                        'channels_first': True,
                        'output_patch_shape': (1,26,26,26),
                        'patch_dimensions': [-3,-2,-1]}

    wholetumor_model = load_old_model(load('Segment_GBM_wholetumor'))
    enhancing_model = load_old_model(load('Segment_GBM_enhancing'))

    wholetumor_prediction = ModelPatchesInference(**wholetumor_prediction_parameters)
    wholetumor_model.append_output([wholetumor_prediction])

    enhancing_prediction = ModelPatchesInference(**enhancing_prediction_parameters)
    enhancing_model.append_output([enhancing_prediction])

    label_binarization = BinarizeLabel(postprocessor_string='_label')

    wholetumor_prediction.append_postprocessor([label_binarization])
    enhancing_prediction.append_postprocessor([label_binarization])

    for case in data_collection.cases:

        print '\nStarting New Case...\n'
        
        wholetumor_file = wholetumor_model.generate_outputs(data_collection, case)[0]['filenames'][-1]

        print wholetumor_file
        data_collection.add_channel(case, wholetumor_file)

        enhancing_file = enhancing_model.generate_outputs(data_collection, case)[0]['filenames'][-1]

    if not save_preprocess:
        for index, file in enumerate(data_collection.data_groups['input_modalities'].preprocessed_case):
            os.remove(file)


if __name__ == '__main__':

    pass