import os

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model
from deepneuro.load.load import load
from deepneuro.preprocessing.preprocessor import Preprocessor, DICOMConverter
from deepneuro.preprocessing.signal import N4BiasCorrection, ZeroMeanNormalization
from deepneuro.preprocessing.transform import Resample, Coregister
from deepneuro.preprocessing.skullstrip import SkullStrip, SkullStrip_Model
from deepneuro.postprocessing.label import BinarizeLabel, LargestComponents, FillHoles
from deepneuro.utilities.util import add_parameter, replace_suffix, quotes, cli_sanitize
from deepneuro.pipelines.shared import load_data, load_model_with_output


def predict_GBM(output_folder, T1POST=None, FLAIR=None, T1PRE=None, ground_truth=None, input_directory=None, bias_corrected=True, resampled=False, registered=False, skullstripped=False, preprocessed=False, save_preprocess=False, save_all_steps=False, output_wholetumor_filename='wholetumor_segmentation.nii.gz', output_enhancing_filename='enhancing_segmentation.nii.gz', verbose=True, input_data=None):

    #--------------------------------------------------------------------#
    # Step 1, Load Data
    #--------------------------------------------------------------------#

    data_collection = load_data(inputs=[T1POST, FLAIR, T1PRE], output_folder=output_folder, input_directory=input_directory, ground_truth=ground_truth, input_data=input_data, verbose=verbose)

    #--------------------------------------------------------------------#
    # Step 2, Preprocess Data
    #--------------------------------------------------------------------#

    if not preprocessed:

        # Random hack to save DICOMs to niftis for further processing.
        preprocessing_steps = [DICOMConverter(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not bias_corrected:
            preprocessing_steps += [N4BiasCorrection(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder)]

        if not registered:
            preprocessing_steps += [Coregister(data_groups=['input_modalities'], save_output=(save_preprocess or save_all_steps), verbose=verbose, output_folder=output_folder, reference_channel=0)]

        if not skullstripped:

            # Skullstripping model included
            skullstripping_prediction_parameters = {'inputs': ['input_modalities'], 
                    'output_filename': os.path.join(output_folder, 'skullstrip_mask.nii.gz'),
                    'batch_size': 50,
                    'patch_overlaps': 3,
                    'channels_first': False,
                    'patch_dimensions': [-4, -3, -2],
                    'output_patch_shape': (56, 56, 6, 1),
                    'save_to_file': False}
            skullstripping_model = load_model_with_output(model_name='skullstrip_mri', outputs=[ModelPatchesInference(**skullstripping_prediction_parameters)], postprocessors=[BinarizeLabel(), FillHoles(), LargestComponents()])

            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, preprocessor_string='_normed')]

            preprocessing_steps += [SkullStrip_Model(data_groups=['input_modalities'], model=skullstripping_model, save_output=save_all_steps, verbose=verbose, output_folder=output_folder, reference_channel=[0, 1])]

            preprocessing_steps += [ZeroMeanNormalization(data_groups=['input_modalities'], save_output=save_all_steps, verbose=verbose, output_folder=output_folder, mask_preprocessor=preprocessing_steps[-1], preprocessor_string='_preprocessed')]

        data_collection.append_preprocessor(preprocessing_steps)

    #--------------------------------------------------------------------#
    # Step 3, Segmentation
    #--------------------------------------------------------------------#

    wholetumor_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, output_wholetumor_filename),
                        'batch_size': 50,
                        'patch_overlaps': 8,
                        'channels_first': False,
                        'patch_dimensions': [-4, -3, -2],
                        'output_patch_shape': (56, 56, 6, 1),
                        'input_channels': [0, 1]}
                        # 'input_channels': [0, 3],}

    enhancing_prediction_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': os.path.join(output_folder, output_enhancing_filename),
                        'batch_size': 50,
                        'patch_overlaps': 8,
                        'channels_first': False,
                        'output_patch_shape': (56, 56, 6, 1),
                        'patch_dimensions': [-4, -3, -2]}

    wholetumor_model = load_model_with_output(model_name='gbm_wholetumor_mri', outputs=[ModelPatchesInference(**wholetumor_prediction_parameters)], postprocessors=[BinarizeLabel(postprocessor_string='_label')])
    enhancing_model = load_model_with_output(model_name='gbm_enhancingtumor_mri', outputs=[ModelPatchesInference(**enhancing_prediction_parameters)], postprocessors=[BinarizeLabel(postprocessor_string='_label')])

    for case in data_collection.cases:

        print '\nStarting New Case...\n'
        
        print 'Whole Tumor Prediction'
        print '======================'
        wholetumor_file = wholetumor_model.generate_outputs(data_collection, case)[0]['filenames'][-1]

        data_collection.add_channel(case, wholetumor_file)

        print 'Enhancing Tumor Prediction'
        print '======================'
        enhancing_file = enhancing_model.generate_outputs(data_collection, case)[0]['filenames'][-1]

        data_collection.clear_outputs()


if __name__ == '__main__':

    pass