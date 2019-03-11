import os

#--------------------------------------------------------------------#
# Step 0, Import DeepNeuro Commands
#--------------------------------------------------------------------#

from deepneuro.outputs import PatchesInference
from deepneuro.preprocessing import DICOMConverter, N4BiasCorrection, ZeroMeanNormalization, Coregister, SkullStrip_Model
from deepneuro.postprocessing import BinarizeLabel, LargestComponents, FillHoles
from deepneuro.pipelines.shared import load_data
from deepneuro.models.model import load_model_with_output
from deepneuro.utilities import docker_print


def predict_brain_mets(output_folder, 
                        T2=None, 
                        T1POST=None, 
                        T1PRE=None, 
                        FLAIR=None, 
                        ground_truth=None, 
                        input_directory=None, 
                        bias_corrected=True,
                        registered=False, 
                        skullstripped=False, 
                        preprocessed=False, 
                        output_segmentation_filename='segmentation.nii.gz',
                        output_probabilities=False, 
                        quiet=False, 
                        input_data=None,
                        save_only_segmentations=False, 
                        save_all_steps=False):

    verbose = not quiet
    save_preprocessed = not save_only_segmentations

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    data_collection = load_data(inputs=[T1PRE, T1POST, T2, FLAIR], output_folder=output_folder, input_directory=input_directory, ground_truth=ground_truth, input_data=input_data, verbose=verbose)

    #--------------------------------------------------------------------#
    # Step 2, Load Models
    #--------------------------------------------------------------------#

    mets_prediction_parameters = {'inputs': ['input_data'], 
                        'output_directory': output_folder,
                        'output_filename': output_segmentation_filename,
                        'batch_size': 50,
                        'patch_overlaps': 8,
                        'output_patch_shape': (28, 28, 28, 1),
                        'output_channels': [1],
                        'case_in_filename': False,
                        'verbose': verbose}

    mets_model = load_model_with_output(model_name='mets_enhancing', outputs=[PatchesInference(**mets_prediction_parameters)], postprocessors=[BinarizeLabel(postprocessor_string='label')], wcc_weights={0: 0.1, 1: 3.0})

    #--------------------------------------------------------------------#
    # Step 3, Add Data Preprocessors
    #--------------------------------------------------------------------#

    if not preprocessed:

        # Random hack to save DICOMs to niftis for further processing.
        preprocessing_steps = [DICOMConverter(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not skullstripped:
            skullstripping_prediction_parameters = {'inputs': ['input_data'], 
                'output_filename': os.path.join(output_folder, 'skullstrip_mask.nii.gz'),
                'batch_size': 50,
                'patch_overlaps': 3,
                'output_patch_shape': (56, 56, 6, 1),
                'save_to_file': False,
                'data_collection': data_collection}

            skullstripping_model = load_model_with_output(model_name='skullstrip_mri', outputs=[PatchesInference(**skullstripping_prediction_parameters)], postprocessors=[BinarizeLabel(), FillHoles(), LargestComponents()])

        if not bias_corrected:
            preprocessing_steps += [N4BiasCorrection(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=1)]

        if not skullstripped:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

            preprocessing_steps += [SkullStrip_Model(data_groups=['input_data'], model=skullstripping_model, save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=[3, 1])]

            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_preprocessed, verbose=verbose, output_folder=output_folder, mask_preprocessor=preprocessing_steps[-1], preprocessor_string='_preprocessed')]

        else:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_preprocessed, verbose=verbose, output_folder=output_folder, mask_zeros=True, preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 4, Run Inference
    #--------------------------------------------------------------------#

    if verbose:
        docker_print('Starting New Case...')
        
        docker_print('Enhancing Mets Prediction')
        docker_print('======================')
    
    mets_model.generate_outputs(data_collection, output_folder)

    data_collection.clear_preprocessor_outputs()


if __name__ == '__main__':

    pass