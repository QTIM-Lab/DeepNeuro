
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model

# Temporary
from keras.utils import plot_model
import glob

def train_Segment_GBM(data_directory, val_data_directory):

    # Define input modalities to load.
    if True:
        training_modality_dict = {'input_modalities': 
    ['*FLAIR_pp.*', '*T2_pp.*', '*T1_pp.*', '*T1post_pp.*'],
    'ground_truth': ['*full_edemamask_pp.*']}
    else:
        training_modality_dict = {'input_modalities': 
        [['*FLAIR_pp.*', 'FLAIR_norm2*'], ['*T1post_pp.*', 'T1post_norm2*']],
        'ground_truth': ['*full_edemamask_pp.*', 'FLAIRmask-label.nii.gz']}

    load_data = True
    train_model = True
    load_test_data = True
    predict = True

    training_data = './wholetumor_predict_patches_test3.h5'
    model_file = 'wholetumor_segnet-58-0.38.h5'
    testing_data = './brats_test_case.h5'

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
            
        # Add patch augmentation
        patch_augmentation = ExtractPatches(patch_shape=(32,32,32), patch_region_conditions=[[brain_region, 1]], data_groups=['input_modalities', 'ground_truth'])
        training_data_collection.append_augmentation(patch_augmentation, multiplier=200)

        # Add left-right flips
        flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities', 'ground_truth'])
        training_data_collection.append_augmentation(flip_augmentation, multiplier=2)

        # Write data to hdf5
        training_data_collection.write_data_to_file(training_data)

    # Or load pre-loaded data.
    training_data_collection = DataCollection(data_storage=training_data, verbose=True)
    training_data_collection.fill_data_groups()

    # Define model parameters
    model_parameters = {'input_shape': (32, 32, 32, 4),
                    'downsize_filters_factor': 1,
                    'pool_size': (2, 2, 2), 
                    'filter_shape': (3, 3, 3), 
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
    if train_model:
        unet_model = UNet(**model_parameters)
        plot_model(unet_model.model, to_file='model_image_dn.png', show_shapes=True)
        training_parameters = {'input_groups': ['input_modalities', 'ground_truth'],
                        'output_model_filepath': 'wholetumor_segnet-{epoch:02d}-{loss:.2f}.h5',
                        'training_batch_size': 2,
                        'num_epochs': 100,
                        'training_steps_per_epoch': 200,
                        'save_best_only': False}
        unet_model.train(training_data_collection, **training_parameters)
    else:
        unet_model = load_old_model(model_file)

    # Load testing data..
    if not os.path.exists(testing_data) or load_test_data:
        # Create a Data Collection
        testing_data_collection = DataCollection(val_data_directory, modality_dict=training_modality_dict, verbose=True)
        testing_data_collection.fill_data_groups()
        # Write data to hdf5
        testing_data_collection.write_data_to_file(testing_data)

    if predict:
        testing_data_collection = DataCollection(data_storage=testing_data, verbose=True)
        testing_data_collection.fill_data_groups()

        testing_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': 'deepneuro.nii.gz',
                        'batch_size': 200,
                        'patch_overlaps': 1}

        prediction = ModelPatchesInference(testing_data_collection, **testing_parameters)

        unet_model.append_output([prediction])
        unet_model.generate_outputs()


if __name__ == '__main__':

    data_directory = ['~/BRATS2017/Train', '~/BRATS2017/Train']
    val_data_directory = ''

    train_Segment_GBM(data_directory, val_data_directory)