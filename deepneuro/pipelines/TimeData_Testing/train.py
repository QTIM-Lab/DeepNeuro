
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model

def train_Segment_GBM(data_directory, val_data_directory):

    # Define input modalities to load.
    # training_modality_dict = {'input_modalities': 
    # ['*FLAIR_pp.*', '*T2_pp.*', '*T1_pp.*', '*T1post_pp.*'],
    # 'ground_truth': ['*full_edemamask_pp.*']}

    training_modality_dict = {'input_modalities': 
    ['*phantom*'],
    'ground_truth': ['*ktrans*']}

    # Write the data to hdf5
    if not os.path.exists('./test_time_small.h5'):

        # Create a Data Collection
        training_data_collection = DataCollection(data_directory, modality_dict=training_modality_dict, verbose=True)
        training_data_collection.fill_data_groups()

        # Add left-right flips
        flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities', 'ground_truth'])
        training_data_collection.append_augmentation(flip_augmentation, multiplier=2)

        # Define patch sampling regions
        def brain_region(data):
            # return (data['ground_truth'] != 1) & (data['input_modalities'] != 0)
            return (data['ground_truth'] >= .1)
        def roi_region(data):
            return data['ground_truth'] == 1

        # Add patch augmentation
        patch_augmentation = ExtractPatches(patch_shape=(3,3,3), patch_region_conditions=[[brain_region, 1]], data_groups=['input_modalities', 'ground_truth'])
        training_data_collection.append_augmentation(patch_augmentation, multiplier=5000)

        # Write data to hdf5
        training_data_collection.write_data_to_file('./test_time_small.h5')
        # training_data_collection.clear_augmentations()

    # Or load pre-loaded data.
    else:
        training_data_collection = DataCollection(data_storage='./test_time_small.h5', verbose=True)
        training_data_collection.fill_data_groups()

    # # Define model parameters
    # model_parameters = {'input_shape': (32, 32, 32, 4),
    #                 'downsize_filters_factor': 3,
    #                 'pool_size': (2, 2, 2), 
    #                 'filter_shape': (3, 3, 3), 
    #                 'dropout': .1, 
    #                 'batch_norm': False, 
    #                 'initial_learning_rate': 0.00001, 
    #                 'output_type': 'binary_label',
    #                 'num_outputs': 1, 
    #                 'activation': 'relu',
    #                 'padding': 'same', 
    #                 'implementation': 'keras',
    #                 'depth': 4,
    #                 'max_filter': 512}

    model_parameters = {'input_shape': (65, 3, 3, 3, 1),
                    'downsize_filters_factor': 1,
                    'pool_size': (2, 2, 2), 
                    'filter_shape': (2, 2, 2), 
                    'dropout': .1, 
                    'batch_norm': True, 
                    'initial_learning_rate': 0.000001, 
                    'output_type': 'regression',
                    'num_outputs': 1, 
                    'activation': 'relu',
                    'padding': 'same', 
                    'implementation': 'keras',
                    'depth': 1,
                    'max_filter': 32}

    # Create U-Net
    try:
    # if True:
        if True:
            unet_model = TimeNet(**model_parameters)
            training_parameters = {'input_groups': ['input_modalities', 'ground_truth'],
                            'output_model_filepath': 'timenet.h5',
                            'training_batch_size': 1000,
                            'num_epochs': 50,
                            'training_steps_per_epoch': 50}

            unet_model.train(training_data_collection, **training_parameters)
        # Or load an old one
        else:
            unet_model = load_old_model('timenet.h5')
    except:
        pass

    # Load testing data..
    if not os.path.exists('./test_val_time_high.h5') and False:
        # Create a Data Collection
        testing_data_collection = DataCollection(val_data_directory, modality_dict=training_modality_dict, verbose=True)
        testing_data_collection.fill_data_groups()

        # Write data to hdf5
        testing_data_collection.write_data_to_file('./test_val_time_high.h5')
    # Or load pre-loaded data.
    else:
        testing_data_collection = DataCollection(data_storage='./test_val_time_high.h5', verbose=True)
        testing_data_collection.fill_data_groups()

    prediction = ModelPatchesInference(testing_data_collection, inputs=['input_modalities'], output_filename='', batch_size=2000, patch_overlaps=1)

    unet_model.append_output([prediction])
    unet_model.generate_outputs()


if __name__ == '__main__':

    # data_directory = ''

    data_directory = ''
    val_data_directory = ''

    train_Segment_GBM(data_directory, val_data_directory)