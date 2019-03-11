#--------------------------------------------------------------------#
# Step 0, Import DeepNeuro Commands
#--------------------------------------------------------------------#

from deepneuro.outputs import PatchesInference
from deepneuro.preprocessing import DICOMConverter, N4BiasCorrection, ZeroMeanNormalization, Coregister
from deepneuro.postprocessing import BinarizeLabel, LargestComponents, FillHoles
from deepneuro.pipelines.shared import load_data
from deepneuro.models.model import load_model_with_output
from deepneuro.utilities import docker_print


def skull_strip(output_folder, 
                T1POST=None, 
                FLAIR=None, 
                ground_truth=None, 
                input_directory=None, 
                bias_corrected=True, 
                registered=False, 
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

    data_collection = load_data(inputs=[FLAIR, T1POST], output_folder=output_folder, input_directory=input_directory, ground_truth=ground_truth, input_data=input_data, verbose=verbose)

    #--------------------------------------------------------------------#
    # Step 2, Load Models
    #--------------------------------------------------------------------#

    skullstripping_prediction_parameters = {'inputs': ['input_data'], 
            'output_directory': output_folder,
            'output_filename': output_segmentation_filename,
            'batch_size': 50,
            'patch_overlaps': 6,
            'output_patch_shape': (56, 56, 6, 1),
            'case_in_filename': False,
            'verbose': verbose}

    skullstripping_model = load_model_with_output(model_name='skullstrip_mri', outputs=[PatchesInference(**skullstripping_prediction_parameters)], postprocessors=[BinarizeLabel(), FillHoles(), LargestComponents(postprocessor_string='label')])

    #--------------------------------------------------------------------#
    # Step 3, Add Data Preprocessors
    #--------------------------------------------------------------------#

    if not preprocessed:

        # Random hack to save DICOMs to niftis for further processing.
        preprocessing_steps = [DICOMConverter(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not bias_corrected:
            preprocessing_steps += [N4BiasCorrection(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=0)]

        preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_preprocessed, verbose=verbose, output_folder=output_folder)]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 4, Run Inference
    #--------------------------------------------------------------------#

    if verbose:
        docker_print('Starting New Case...')
        
        docker_print('Skullstripping Prediction')
        docker_print('======================')
    
    skullstripping_model.generate_outputs(data_collection, output_folder)

    data_collection.clear_preprocessor_outputs()


if __name__ == '__main__':

    pass