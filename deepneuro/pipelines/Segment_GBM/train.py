
from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches


def train_Segment_GBM(data_directory):

    modality_dict = {'input_modalities': 
                        ['T2.nii.gz', 
                        'T1.nii.gz', 
                        'T1post.nii.gz', 
                        'FLAIR.nii.gz'], 
                    'ground_truth': 
                        ['groundtruth-label.nii.gz']}

    modality_dict = {'input_modalities': ['*FLAIR_pp.*', '*T2_pp.*', '*T1_pp.*', '*T1post_pp.*', '*full_edemamask_pp.*'],
                                        'ground_truth': ['*full_edemamask_pp.*']}

    # Create a Data Collection
    training_data_collection = DataCollection(data_directory, modality_dict)
    training_data_collection.fill_data_groups()

    # # Specify Augmentations
    flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities', 'ground_truth'], multiplier=2)
    # patch_augmentation = ExtractPatches(patch_shape=(32,32,32), patch_extraction_conditions=[], data_groups=['input_modalities', 'ground_truth'], multiplier=2)

    # # Append Augmentations to Data
    for augmentation in [flip_augmentation]:
        training_data_collection.append_augmentation(augmentation)


    # training_data_collection.append_augmentation(flip_augmentation_group)
    training_data_collection.write_data_to_file()

    # if config['validation_dir'] is not None and config['hdf5_validation'] is not None:
    #     # Validation - with patch augmentation
    #     validation_data_collection = DataCollection(config['validation_dir'], modality_dict, brainmask_dir=config['brain_mask_dir'], roimask_dir=config['roi_mask_dir'], patch_shape=config['patch_shape'])
    #     validation_data_collection.fill_data_groups()

    #     if not config['perpetual_patches']:
    #         validation_patch_augmentation = ExtractPatches(config['patch_shape'], config['patch_extraction_conditions'])
    #         validation_patch_augmentation_group = AugmentationGroup({'input_modalities': validation_patch_augmentation, 'ground_truth': validation_patch_augmentation}, multiplier=config['validation_patches_multiplier'])
    #         validation_data_collection.append_augmentation(validation_patch_augmentation_group)
        
    #     validation_data_collection.append_augmentation(flip_augmentation_group)
    #     validation_data_collection.write_data_to_file(output_filepath = config['hdf5_validation'], save_masks=config["overwrite_masks"], store_masks=True)
    # else:
    #     print 'Validation data not available, training without validation data.'

def nested_yield_1():

    gen = nested_yield_2()

    x = 0
    while x < 100:
        print 'start of gen'
        print next(gen)
        x += 1

def nested_yield_2():

    gen = nested_yield_3()

    while True:
        print 'top of yield 2'
        yield next(gen)
        print 'bottom of yield 2'

def nested_yield_3():

    x = 0
    while True:
        print 'top of yield 3'
        yield x
        print 'bottom of yield 3'
        x = x + 1

    return

if __name__ == '__main__':

    # nested_yield_1()

    data_directory = '/mnt/jk489/sharedfolder/BRATS2017/Train'

    train_Segment_GBM(data_directory)