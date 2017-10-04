
from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import Flip_Rotate_2D, ExtractPatches
from deepneuro.models.unet import UNet

def train_Segment_GBM(data_directory):

    # Define input modalities to load.
    training_modality_dict = {'input_modalities': 
    ['*FLAIR_pp.*', '*T2_pp.*', '*T1_pp.*', '*T1post_pp.*', '*full_edemamask_pp.*'],
    'ground_truth': ['*full_edemamask_pp.*']}

    # Create a Data Collection
    training_data_collection = DataCollection(data_directory, training_modality_dict, verbose=True)
    training_data_collection.fill_data_groups()

    # Add left-right flips
    flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False, data_groups=['input_modalities', 'ground_truth'])
    training_data_collection.append_augmentation(flip_augmentation, multiplier=2)

    # Define patch sampling regions
    def brain_region(data):
        return (data['ground_truth'] != 1) & (data['input_modalities'] != 0)
    def roi_region(data):
        return data['ground_truth'] == 1

    # Add patch augmentation
    patch_augmentation = ExtractPatches(patch_shape=(32,32,32), patch_region_conditions=[[brain_region, .3], [roi_region, .7]], data_groups=['input_modalities', 'ground_truth'])
    training_data_collection.append_augmentation(patch_augmentation, multiplier=70)

    # Write the data to hdf5
    training_data_collection.write_data_to_file('./test.h5')

    # Define model parameters
    model_parameters = {input_shape: (32, 32, 32, 4),
                    downsize_filters_factor: 1,
                    pool_size: (2, 2, 2), 
                    filter_shape: (3, 3, 3), 
                    dropout: .1, 
                    batch_norm: False, 
                    initial_learning_rate: 0.00001, 
                    output_type: 'binary_label',
                    num_outputs: 1, 
                    activation: 'relu',
                    padding: 'same', 
                    implementation: 'keras',
                    depth: 4,
                    max_filter=512}

    # Create U-Net
    if True:
        unet_model = UNet(**model_parameters)

    # Or load an old one
    else:
        unet_model = load_old_model('model.h5')

    # Define training parameters
    training_parameters = {}

    # Define training generators
    training_generator = None

    

    # # Create a new model if necessary. Preferably, load an existing one.
    # if not config["overwrite_model"] and os.path.exists(config["model_file"]):
    #     print 'Loading old model...'
    #     model = load_old_model(config["model_file"])
    # else:
    #     model = u_net_3d(input_shape=(len(modality_dict['input_modalities']),) + config['patch_shape'], output_shape=(len(modality_dict['ground_truth']),) + config['patch_shape'], downsize_filters_factor=config['downsize_filters_factor'], initial_learning_rate=config['initial_learning_rate'], regression=config['regression'], num_outputs=(len(modality_dict['ground_truth'])))



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