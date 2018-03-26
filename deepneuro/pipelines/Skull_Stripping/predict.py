import os

from deepneuro.data.data_collection import DataCollection
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model
from deepneuro.load.load import load
from deepneuro.preprocessing.preprocessor import Preprocessor
from deepneuro.preprocessing.signal import N4BiasCorrection, ZeroMeanNormalization
from deepneuro.preprocessing.transform import Resample, Coregister
from deepneuro.postprocessing.label import BinarizeLabel, LargestComponents, FillHoles

def skull_strip(output_folder, T1POST=None, FLAIR=None, ground_truth=None, input_directory=None, bias_corrected=True, resampled=False, registered=False, normalized=False, preprocessed=False, save_preprocess=False, save_all_steps=False, mask_output='skullstrip_mask.nii.gz', verbose=True):

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    input_data = {'input_modalities': [FLAIR, T1POST]}

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
            preprocessing_steps += [Coregister(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=0)]

        if not normalized:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 3, Skullstripping
    #--------------------------------------------------------------------#

    skullstrip_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, mask_output),
                        'batch_size': 25,
                        'patch_overlaps': 8,
                        'channels_first': True,
                        'patch_dimensions': [-3, -2, -1],
                        'output_patch_shape': (1, 64, 64, 32),
                        # 'input_channels': [0, 3],
                        }

    skull_stripping_model = load_old_model(load('Skull_Strip_T1Post_FLAIR'))

    skull_stripping_prediction = ModelPatchesInference(**skullstrip_prediction_parameters)

    label_binarization = BinarizeLabel()
    largest_component = LargestComponents()
    hole_filler = FillHoles(postprocessor_string='_mask')

    skull_stripping_prediction.append_postprocessor([label_binarization, largest_component, hole_filler])

    skull_stripping_model.append_output([skull_stripping_prediction])

    for case in data_collection.cases:

        print '\nStarting New Case...\n'
        
        skull_stripping_prediction.case = case
        skull_stripping_mask = skull_stripping_model.generate_outputs(data_collection)[0]

        print len(skull_stripping_mask)
        for item in skull_stripping_mask:
            print item
        # print 'Finished...', skull_stripping_mask

    if not save_preprocess:
        for index, file in enumerate(data_collection.data_groups['input_modalities'].preprocessed_case):
            os.remove(file)


if __name__ == '__main__':

    # skull_strip pipeline -T1POST /qtim2/users/data/BAV/ANALYSIS/COREGISTRATION/BAV_01/VISIT_01/BAV_01-VISIT_01-MPRAGE_POST_r_T2.nii.gz -FLAIR /qtim2/users/data/BAV/ANALYSIS/COREGISTRATION/BAV_01/VISIT_01/BAV_01-VISIT_01-FLAIR_r_T2.nii.gz -output_folder /home/abeers/Junk/DEEPNEURO_TEST -gpu_num 0 -debiased -registered -resampled

    pass