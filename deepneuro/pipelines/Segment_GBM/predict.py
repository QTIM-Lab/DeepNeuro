import os

#--------------------------------------------------------------------#
# Step 0, Import DeepNeuro Commands
#--------------------------------------------------------------------#

from deepneuro.outputs.segmentation import PatchesInference
from deepneuro.preprocessing.preprocessor import DICOMConverter
from deepneuro.preprocessing.signal import N4BiasCorrection, ZeroMeanNormalization
from deepneuro.preprocessing.transform import Coregister
from deepneuro.preprocessing.skullstrip import SkullStrip_Model
from deepneuro.postprocessing.label import BinarizeLabel, LargestComponents, FillHoles
from deepneuro.pipelines.shared import load_data
from deepneuro.models.model import load_model_with_output
from deepneuro.utilities.util import docker_print


def predict_GBM(output_folder, 
                T1POST=None, 
                FLAIR=None, 
                T1PRE=None, 
                ground_truth=None, 
                input_directory=None, 
                bias_corrected=True, 
                resampled=False, 
                registered=False, 
                skullstripped=False, 
                preprocessed=False, 
                save_only_segmentations=False, 
                save_all_steps=False, 
                output_wholetumor_filename='wholetumor_segmentation.nii.gz', 
                output_enhancing_filename='enhancing_segmentation.nii.gz', 
                output_probabilities=False, 
                quiet=False, 
                input_data=None, 
                registration_reference='FLAIR'):

    verbose = not quiet
    save_preprocessed = not save_only_segmentations

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    data_collection = load_data(inputs=[FLAIR, T1POST, T1PRE], output_folder=output_folder, input_directory=input_directory, ground_truth=ground_truth, input_data=input_data, verbose=verbose)

    #--------------------------------------------------------------------#
    # Step 2, Load Models and Postprocessors
    #--------------------------------------------------------------------#

    wholetumor_prediction_parameters = {'output_directory': output_folder,
                        'output_filename': output_wholetumor_filename,
                        'batch_size': 50,
                        'patch_overlaps': 6,
                        'output_patch_shape': (56, 56, 6, 1),
                        'input_channels': [0, 1],
                        'case_in_filename': False,
                        'verbose': verbose}

    enhancing_prediction_parameters = {'output_directory': output_folder,
                        'output_filename': output_enhancing_filename,
                        'batch_size': 50,
                        'patch_overlaps': 6,
                        'output_patch_shape': (56, 56, 6, 1),
                        'case_in_filename': False,
                        'verbose': verbose}

    wholetumor_model = load_model_with_output(model_name='gbm_wholetumor_mri', 
        outputs=[PatchesInference(**wholetumor_prediction_parameters)], 
        postprocessors=[BinarizeLabel(postprocessor_string='label')])

    enhancing_model = load_model_with_output(model_name='gbm_enhancingtumor_mri', 
        outputs=[PatchesInference(**enhancing_prediction_parameters)], 
        postprocessors=[BinarizeLabel(postprocessor_string='label')])

    #--------------------------------------------------------------------#
    # Step 3, Add Data Preprocessors
    #--------------------------------------------------------------------#

    if not preprocessed:

        preprocessing_steps = [DICOMConverter(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not bias_corrected:
            preprocessing_steps += [N4BiasCorrection(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=0)]

        if not skullstripped:

            skullstripping_prediction_parameters = {'inputs': ['input_data'], 
                'output_filename': os.path.join(output_folder, 'skullstrip_mask.nii.gz'),
                'batch_size': 50,
                'patch_overlaps': 3,
                'output_patch_shape': (56, 56, 6, 1),
                'save_to_file': False,
                'data_collection': data_collection}

            skullstripping_model = load_model_with_output(model_name='skullstrip_mri', outputs=[PatchesInference(**skullstripping_prediction_parameters)], postprocessors=[BinarizeLabel(), FillHoles(), LargestComponents()])

            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

            preprocessing_steps += [SkullStrip_Model(data_groups=['input_data'], model=skullstripping_model, save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=[0, 1])]

            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_preprocessed, verbose=verbose, output_folder=output_folder, mask_preprocessor=preprocessing_steps[-1], preprocessor_string='_preprocessed')]

        else:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_preprocessed, verbose=verbose, output_folder=output_folder, mask_zeros=True, preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 4, Run Inference
    #--------------------------------------------------------------------#

    if verbose:
        docker_print('Starting New Case...')
    
        docker_print('Whole Tumor Prediction')
        docker_print('======================')

    wholetumor_file = wholetumor_model.generate_outputs(data_collection, output_folder)[0]['filenames'][-1]
    data_collection.add_channel(output_folder, wholetumor_file)

    if verbose:
        docker_print('Enhancing Tumor Prediction')
        docker_print('======================')

    enhancing_model.generate_outputs(data_collection, output_folder)

    data_collection.clear_preprocessor_outputs()


if __name__ == '__main__':

    pass