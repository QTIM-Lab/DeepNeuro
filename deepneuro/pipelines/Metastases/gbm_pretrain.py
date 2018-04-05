
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches, MaskData, Downsample, Copy
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model
from deepneuro.postprocessing.label import BinarizeLabel, LargestComponents, FillHoles

# Temporary
from keras.utils import plot_model
import glob

def train_Segment_GBM(data_directory, val_data_directory):

    # Define input modalities to load.
    training_modality_dict = {'input_modalities': 
    ['*FLAIR_pp.*', '*T2_pp.*', '*T1_pp.*', '*T1post_pp.*'],
    'ground_truth': ['enhancingmask_pp.nii.gz']}

    load_data = True
    train_model = True
    load_test_data = False
    predict = True

    training_data = '/mnt/jk489/QTIM_Databank/DeepNeuro_Datasets/BRATS_enhancing_prediction_only_data.h5'
    model_file = '/mnt/jk489/QTIM_Databank/DeepNeuro_Datasets/BRATS_enhancing_prediction_only_model.h5'
    testing_data = '/mnt/jk489/QTIM_Databank/DeepNeuro_Datasets/BRATS_enhancing_prediction_only_data.h5'

    # Write the data to hdf5
    if (not os.path.exists(training_data) and train_model) or load_data:

        # Create a Data Collection
        training_data_collection = DataCollection(data_directory, modality_dict=training_modality_dict, verbose=True)
        training_data_collection.fill_data_groups()

        # Define patch sampling regions
        def brain_region(data):
            return (data['ground_truth'] != 1) & (data['input_modalities'] != 0)
        def roi_region(data):
            return data['ground_truth'] == 1
        def empty_region(data):
            return data['input_modalities'] == 0

        # Add patch augmentation
        patch_augmentation = ExtractPatches(patch_shape=(32, 32, 32), patch_region_conditions=[[empty_region, .05], [brain_region, .25], [roi_region, .7]], data_groups=['input_modalities', 'ground_truth'], patch_dimensions={'ground_truth': [0, 1, 2], 'input_modalities': [0, 1, 2]})
        training_data_collection.append_augmentation(patch_augmentation, multiplier=2000)

        # Write data to hdf5
        training_data_collection.write_data_to_file(training_data)

    if train_model:
        # Or load pre-loaded data.
        training_data_collection = DataCollection(data_storage=training_data, verbose=True)
        training_data_collection.fill_data_groups()

        # Add left-right flips
        flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities', 'ground_truth'])
        training_data_collection.append_augmentation(flip_augmentation, multiplier=2)

        # Define model parameters
        model_parameters = {'input_shape': (32, 32, 32, 4),
                        'downsize_filters_factor': 1,
                        'pool_size': (2, 2, 2), 
                        'filter_shape': (5, 5, 5), 
                        'dropout': 0, 
                        'batch_norm': True, 
                        'initial_learning_rate': 0.000001, 
                        'output_type': 'binary_label',
                        'num_outputs': 1, 
                        'activation': 'relu',
                        'padding': 'same', 
                        'implementation': 'keras',
                        'depth': 4,
                        'max_filter': 512}

        # Create U-Net
        unet_model = UNet(**model_parameters)
        plot_model(unet_model.model, to_file='model_image_dn.png', show_shapes=True)
        training_parameters = {'input_groups': ['input_modalities', 'ground_truth'],
                        'output_model_filepath': model_file,
                        'training_batch_size': 64,
                        'num_epochs': 1000,
                        'training_steps_per_epoch': 20}
        unet_model.train(training_data_collection, **training_parameters)
    else:
        unet_model = load_old_model(model_file)

    if predict:
        testing_data_collection = DataCollection(val_data_directory, modality_dict=testing_modality_dict, verbose=True)
        testing_data_collection.fill_data_groups()

        if load_test_data:
            # Write data to hdf5
            testing_data_collection.write_data_to_file(testing_data)

        testing_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': 'mets_prediction.nii.gz',
                        'batch_size': 250,
                        'patch_overlaps': 1,
                        'output_patch_shape': (26, 26, 26, 4),
                        'save_all_steps': True}

        prediction = ModelPatchesInference(**testing_parameters)

        label_binarization = BinarizeLabel()
        largest_component = LargestComponents()
        hole_filler = FillHoles(postprocessor_string='_label')

        prediction.append_postprocessor([label_binarization, largest_component, hole_filler])

        unet_model.append_output([prediction])
        unet_model.generate_outputs(testing_data_collection)


if __name__ == '__main__':

    data_directory = ['/mnt/jk489/sharedfolder/BRATS2017/Train', '/mnt/jk489/sharedfolder/BRATS2017/Val']
    val_data_directory = ['/mnt/jk489/QTIM_Databank/QTIM_CLINICAL/Preprocessed/PEM/VAL', '/mnt/jk489/QTIM_Databank/QTIM_CLINICAL/Preprocessed/BRE/VAL']

    train_Segment_GBM(data_directory, val_data_directory)