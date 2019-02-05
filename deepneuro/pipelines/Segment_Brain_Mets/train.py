import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from deepneuro.data.data_collection import DataCollection
from deepneuro.augmentation.augment import ExtractPatches
from deepneuro.postprocessing.label import BinarizeLabel
from deepneuro.preprocessing.signal import ZeroMeanNormalization
from deepneuro.models.weighted_cat_cross_entropy import WeightedCategoricalCrossEntropy

TrainingDataCollection = DataCollection(data_sources={'csv': 'Metastases_Data_Train.csv'})
TestingDataCollection = DataCollection(data_sources={'csv': 'Metastases_Data_Test.csv'})

Normalization = ZeroMeanNormalization(data_groups=['input_data'])
TrainingDataCollection.append_preprocessor(Normalization)

def BrainRegion(data):
    return data['input_data'] != 0
def TumorRegion(data):
    return data['ground_truth'] == 1

PatchAugmentation = ExtractPatches(patch_shape=(32, 32, 32), 
    patch_region_conditions=[[BrainRegion, 0.70], [TumorRegion, 0.30]])
TrainingDataCollection.append_augmentation(PatchAugmentation, multiplier=20)
TrainingDataCollection.write_data_to_file('training_data.hdf5')

ModelParameters = {'input_shape': (32, 32, 32, 1),
                'cost_function': 'weighted_categorical_label',
                ''}
UNETModel = UNet(**ModelParameters)

TrainingParameters = {'output_model_filepath': 'unet_metastases.py',
                'training_batch_size': 16,
                'num_epochs': 50}
UNETModel.train(TrainingDataCollection, **TrainingParameters)

TestingParameters = {'inputs': ['input_data'],
               'output_filename': '_segmentation.nii.gz',
               'batch_size': 20,
               'output_patch_shape': (30, 30, 30, 1),
               'patch_overlaps': 4,
               'output_directory': './Prediction_Outputs'}
Prediction = ModelPatchesInference(**TestingParameters)

LabelBinarization = BinarizeLabel(binarization_threshold=.5)
ErrorStatistics = ErrorCalculation(output_log='inference_statistics.csv', 
    cost_functions=['dice', 'cluster_accuracy'])
Prediction.append_postprocessor([LabelBinarization, ErrorStatistics])

UNETModel.append_output([Prediction])
UNETModel.generate_outputs(TestingDataCollection)