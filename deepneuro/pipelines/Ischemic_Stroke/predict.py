from deepneuro.outputs import PatchesInference
from deepneuro.preprocessing.preprocessor import DICOMConverter
from deepneuro.preprocessing.signal import ZeroMeanNormalization
from deepneuro.preprocessing.transform import Coregister
from deepneuro.postprocessing.label import BinarizeLabel
from deepneuro.pipelines.shared import load_data 
from deepneuro.models.model import load_model_with_output
from deepneuro.utilities.util import docker_print


def predict_ischemic_stroke(output_folder, 
                            B0, 
                            DWI, 
                            ground_truth=None, 
                            input_directory=None,
                            registered=False,
                            preprocessed=False, 
                            save_only_segmentations=False, 
                            save_all_steps=False, 
                            output_segmentation_filename='segmentation.nii.gz',
                            input_data=None, 
                            registration_reference='FLAIR',
                            quiet=False):

    verbose = not quiet
    save_preprocessed = not save_only_segmentations
    registration_reference_channel = 1

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    data_collection = load_data(inputs=[B0, DWI], output_folder=output_folder, input_directory=input_directory, ground_truth=ground_truth, input_data=input_data, verbose=verbose)

    #--------------------------------------------------------------------#
    # Step 2, Load Models
    #--------------------------------------------------------------------#

    stroke_prediction_parameters = {'inputs': ['input_data'], 
                        'output_directory': output_folder,
                        'output_filename': output_segmentation_filename,
                        'batch_size': 50,
                        'patch_overlaps': 6,
                        'output_patch_shape': (62, 62, 6, 1),
                        'case_in_filename': False,
                        'verbose': verbose}

    stroke_model = load_model_with_output(model_name='ischemic_stroke', outputs=[PatchesInference(**stroke_prediction_parameters)], postprocessors=[BinarizeLabel(postprocessor_string='label')])

    #--------------------------------------------------------------------#
    # Step 3, Add Data Preprocessors
    #--------------------------------------------------------------------#

    if not preprocessed:

        preprocessing_steps = [DICOMConverter(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_data'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=registration_reference_channel)]

        if not preprocessed:
            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_data'], save_output=save_preprocessed, verbose=verbose, output_folder=output_folder, mask_zeros=True, preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 4, Run Inference
    #--------------------------------------------------------------------#

    if verbose:
        docker_print('Starting New Case...')
        
        docker_print('Ischemic Stroke Prediction')
        docker_print('======================')
    
    stroke_model.generate_outputs(data_collection, output_folder)

    data_collection.clear_preprocessor_outputs()


if __name__ == '__main__':

    pass