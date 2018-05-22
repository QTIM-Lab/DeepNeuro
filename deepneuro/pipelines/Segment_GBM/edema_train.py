
import os
import glob

# GPU NUMBER GOES HERE!
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches, MaskData, Downsample, Copy, Flip_Rotate_3D, Shift_Squeeze_Intensities
from deepneuro.models.unet import UNet
from deepneuro.models.timenet import TimeNet
from deepneuro.outputs.inference import ModelPatchesInference
from deepneuro.models.model import load_old_model
from deepneuro.postprocessing.label import BinarizeLabel, LargestComponents, FillHoles

# Temporary
from keras.utils import plot_model

def train_Segment_GBM(data_directory, val_data_directory):

    # Define input modalities to load.
    training_modality_dict = {'input_modalities': 
    [['FLAIR_pp.*', 'FLAIR_norm2.nii.gz', 'FLAIR_r_T2.nii.gz'], ['T2_pp.*', 'T2_norm2.nii.gz', 'T2SPACE.nii.gz'], ['T1post_pp.*', 'T1post_norm2.nii.gz', 'T1Post_r_T2.nii.gz'], ['T1_pp.*', 'T1_norm2.nii.gz', 'T1Pre_r_T2.nii.gz']],
    'ground_truth': [['edemamask_pp.nii.gz', 'FLAIRmask-label.nii.gz']]}

    load_data = False
    train_model = False
    load_test_data = False
    predict = True

    training_data = '/mnt/jk489/QTIM_Databank/DeepNeuro_Datasets/BRATS_multi_institution_edema.h5'
    model_file = '/mnt/jk489/QTIM_Databank/DeepNeuro_Datasets/BRATS_multi_institution_edema_model_modality_drop.h5'
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
        patch_augmentation = ExtractPatches(patch_shape=(32, 32, 32), 
            patch_region_conditions=[[empty_region, .05], [brain_region, .25], [roi_region, .7]],
            data_groups=['input_modalities', 'ground_truth'], 
            patch_dimensions={'ground_truth': [1, 2, 3], 'input_modalities': [1, 2, 3]})
        training_data_collection.append_augmentation(patch_augmentation, multiplier=200)

        # Write data to hdf5
        training_data_collection.write_data_to_file(training_data)

    if train_model:
        # Or load pre-loaded data.
        training_data_collection = DataCollection(data_storage=training_data, verbose=True)
        training_data_collection.fill_data_groups()

        # Add left-right flips
        # flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities', 'ground_truth'])
        flip_augmentation = Flip_Rotate_3D(data_groups=['input_modalities', 'ground_truth'])
        downsample_augmentation = MaskData(mask_channels={'input_modalities': [1, 2, 3]}, random_sample=False, data_groups=['input_modalities'])
        mask_augmentation = Downsample(channel=0, axes={'input_modalities': [1, 2, 3]}, factor=2, random_sample=False, data_groups=['input_modalities'])
        itensity_augmentation = Shift_Squeeze_Intensities(data_groups=['input_modalities'])
        training_data_collection.append_augmentation(downsample_augmentation, multiplier=3)
        training_data_collection.append_augmentation(mask_augmentation, multiplier=3)
        training_data_collection.append_augmentation(flip_augmentation, multiplier=2)
        training_data_collection.append_augmentation(itensity_augmentation, multiplier=4)

        # Define model parameters
        model_parameters = {'input_shape': (32, 32, 32, 4),
                        'downsize_filters_factor': 1,
                        'pool_size': (2, 2, 2), 
                        'filter_shape': (5, 5, 5), 
                        'dropout': 0, 
                        'batch_norm': True, 
                        'initial_learning_rate': 0.0001, 
                        'output_type': 'binary_label',
                        'num_outputs': 1, 
                        'activation': 'relu',
                        'padding': 'same', 
                        'implementation': 'keras',
                        'depth': 4,
                        'max_filter': 256}

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

    # Define input modalities to load.
    testing_modality_dict = {'input_modalities': 
    [['FLAIR_pp.*', 'FLAIR_norm2.nii.gz', 'FLAIR_r_T2.nii.gz'], ['T2_pp.*', 'T2_norm2.nii.gz', 'T2SPACE.nii.gz'], ['T1post_pp.*', 'T1post_norm2.nii.gz', 'T1Post_r_T2.nii.gz'], ['T1_pp.*', 'T1_norm2.nii.gz', 'T1Pre_r_T2.nii.gz']]}

    if predict:
        testing_data_collection = DataCollection(val_data_directory, modality_dict=testing_modality_dict, verbose=True)
        testing_data_collection.fill_data_groups()

        if load_test_data:
            # Write data to hdf5
            testing_data_collection.write_data_to_file(testing_data)

        testing_parameters = {'inputs': ['input_modalities'], 
                        'output_filename': 'multi_brats_edema_missing_modality.nii.gz',
                        'batch_size': 50,
                        'patch_overlaps': 8,
                        'output_patch_shape': (26, 26, 26, 4),
                        'save_all_steps': True}

        prediction = ModelPatchesInference(**testing_parameters)

        label_binarization = BinarizeLabel(postprocessor_string='_label')

        prediction.append_postprocessor([label_binarization])

        unet_model.append_output([prediction])
        unet_model.generate_outputs(testing_data_collection)


if __name__ == '__main__':

    data_directory = ['/mnt/jk489/sharedfolder/BRATS2017/Train', '/mnt/jk489/sharedfolder/segmentation_2/patients', '/mnt/jk489/sharedfolder/BRATS2017/Val']
    val_data_directory = ['/mnt/jk489/QTIM_Databank/QTIM_CLINICAL/Preprocessed/TMZ']

    train_Segment_GBM(data_directory, val_data_directory)