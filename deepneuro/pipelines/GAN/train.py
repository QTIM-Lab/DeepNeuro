
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches, MaskData, Downsample, Copy
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.models.gan import GAN
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model

# Temporary
from keras.utils import plot_model
import glob

def train_Segment_GBM(data_directory, val_data_directory):

    # Define input modalities to load.
    training_modality_dict = {'input_modalities': [['T1_pp*']]}

    load_data = False
    train_model = True
    load_test_data = False
    predict = False

    training_data = '/mnt/jk489/QTIM_Databank/DeepNeuro_Datasets/GAN_data_corrected.h5'
    model_file = 'GAN_model.h5'
    testing_data = './GAN_test.h5'

    # Write the data to hdf5
    if (not os.path.exists(training_data) and train_model) or load_data:

        # Create a Data Collection
        training_data_collection = DataCollection(data_directory, modality_dict=training_modality_dict, verbose=True)
        training_data_collection.fill_data_groups()

        # Define patch sampling regions
        def brain_region(data):
            # return (data['ground_truth'] != 1) & (data['input_modalities'] != 0)
            return data['input_modalities'] != 0
        def roi_region(data):
            return data['ground_truth'] == 1

        # Add patch augmentation
        patch_augmentation = ExtractPatches(patch_shape=(64, 64, 8), patch_region_conditions=[[brain_region, 1]], data_groups=['input_modalities'], patch_dimensions={'input_modalities': [0,1,2]})
        training_data_collection.append_augmentation(patch_augmentation, multiplier=200)

        # Write data to hdf5
        training_data_collection.write_data_to_file(training_data)

    if train_model:
        # Or load pre-loaded data.
        training_data_collection = DataCollection(data_storage=training_data, verbose=True)
        training_data_collection.fill_data_groups()

        # Add left-right flips
        flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities'])
        training_data_collection.append_augmentation(flip_augmentation, multiplier=2)

        # Define model parameters
        model_parameters = {'input_shape': (32, 32, 32, 4),
                        'downsize_filters_factor': 1,
                        'pool_size': (2, 2, 2), 
                        'filter_shape': (5, 5, 5), 
                        'dropout': 0.5, 
                        'batch_norm': True, 
                        'initial_learning_rate': 0.0001, 
                        'output_type': 'regression',
                        'num_outputs': 1, 
                        'activation': 'relu',
                        'padding': 'same', 
                        'implementation': 'keras',
                        'depth': 4,
                        'max_filter': 256}

        # Create U-Net
        GAN_model = GAN(**model_parameters)
        # plot_model(GAN_model.model, to_file='model_image_dn.png', show_shapes=True)
        training_parameters = {'input_groups': ['input_modalities'],
                        'output_model_filepath': model_file,
                        'training_batch_size': 32,
                        'num_epochs': 10000,
                        'training_steps_per_epoch': 20}
        GAN_model.train(training_data_collection, **training_parameters)
    else:
        GAN_model = load_old_model('DCGAN_150.model.meta', backend='tf')

    if predict:
        print GAN_model
        for i in GAN_model.graph.get_operations():
            print i
        # print GAN_model.run('generator')
        # testing_parameters = {'inputs': ['input_modalities'], 
        #                 'output_filename': 'deepneuro.nii.gz',
        #                 'batch_size': 250,
        #                 'patch_overlaps': 6,
        #                 'output_patch_shape': (26,26,26,4)}

        # prediction = ModelPatchesInference(testing_data_collection, **testing_parameters)

        # unet_model.append_output([prediction])
        # unet_model.generate_outputs()


if __name__ == '__main__':

    data_directory = '/mnt/jk489/QTIM_Databank/QTIM_CLINICAL/Preprocessed/Train'
    val_data_directory = '/mnt/jk489/QTIM_Databank/QTIM_CLINICAL/Preprocessed/Test_Case'

    train_Segment_GBM(data_directory, val_data_directory)