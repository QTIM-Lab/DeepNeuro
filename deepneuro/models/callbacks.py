import keras
import os
import imageio
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from deepneuro.utilities.util import add_parameter
from deepneuro.utilities.visualize import check_data


class EpochPredict(keras.callbacks.Callback):

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'epoch_prediction_data_collection', self.data_collection)
        add_parameter(self, kwargs, 'epoch_prediction_object', None)
        add_parameter(self, kwargs, 'deepneuro_model', None)
        add_parameter(self, kwargs, 'epoch_prediction_dir', None)
        add_parameter(self, kwargs, 'output_gif', None)
        add_parameter(self, kwargs, 'batch_size', 1)
        add_parameter(self, kwargs, 'epoch_prediction_batch_size', self.batch_size)

        if not os.path.exists(self.epoch_prediction_dir):
            os.mkdir(self.epoch_prediction_dir)

        self.predictions = []

        # There's a more concise way to do this..
        self.predict_data = next(self.epoch_prediction_data_collection.data_generator(perpetual=True, verbose=False, just_one_batch=True, batch_size=self.epoch_prediction_batch_size))
 
    def on_train_end(self, logs={}):
        imageio.mimsave(os.path.join(self.epoch_prediction_dir, 'epoch_prediction.gif'), self.predictions)
        return
 
    def on_epoch_end(self, epoch, logs={}):

        if self.epoch_prediction_object is None:
            prediction = self.deepneuro_model.predict(self.predict_data[self.deepneuro_model.input_data])
        else:
            prediction = self.epoch_prediction_object.process_case(self.predict_data[self.deepneuro_model.input_data], model=self.deepneuro_model)

        output_filepaths, output_images = check_data({'prediction': prediction}, output_filepath=os.path.join(self.epoch_prediction_dir, 'epoch_{}.png'.format(epoch)), show_output=False, batch_size=self.epoch_prediction_batch_size)

        self.predictions += [output_images['prediction'].astype('uint8')]

        return


class GANPredict(keras.callbacks.Callback):

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'epoch_prediction_data_collection', self.data_collection)
        add_parameter(self, kwargs, 'epoch_prediction_object', None)
        add_parameter(self, kwargs, 'deepneuro_model', None)
        add_parameter(self, kwargs, 'epoch_prediction_dir', None)
        add_parameter(self, kwargs, 'output_gif', None)
        add_parameter(self, kwargs, 'batch_size', 1)
        add_parameter(self, kwargs, 'epoch_prediction_batch_size', self.batch_size)
        add_parameter(self, kwargs, 'latent_size', 128)
        add_parameter(self, kwargs, 'sample_latent', np.random.normal(size=[self.epoch_prediction_batch_size, self.latent_size]))

        if not os.path.exists(self.epoch_prediction_dir):
            os.mkdir(self.epoch_prediction_dir)

        self.predictions = []

    def on_train_end(self, logs={}):
        imageio.mimsave(os.path.join(self.epoch_prediction_dir, 'epoch_prediction.gif'), self.predictions)
        return
 
    def on_epoch_end(self, epoch, logs={}):

        if self.epoch_prediction_object is None:
            prediction = self.deepneuro_model.predict(sample_latent=self.sample_latent)
        else:
            prediction = self.epoch_prediction_object.process_case(self.predict_data[self.deepneuro_model.input_data], model=self.deepneuro_model)

        output_filepaths, output_images = check_data({'prediction': prediction}, output_filepath=os.path.join(self.epoch_prediction_dir, 'epoch_{}.png'.format(epoch)), show_output=False, batch_size=self.epoch_prediction_batch_size)

        self.predictions += [output_images['prediction'].astype('uint8')]

        return


class SaveModel(keras.callbacks.Callback):

    def __init__(self, **kwargs):

        # Add save best only.

        add_parameter(self, kwargs, 'deepneuro_model', None)

    def on_train_begin(self, logs={}):
        self.deepneuro_model.save_model(self.deepneuro_model.output_model_filepath)
        return

    def on_train_end(self, logs={}):
        self.deepneuro_model.save_model(self.deepneuro_model.output_model_filepath)
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.deepneuro_model.save_model(self.deepneuro_model.output_model_filepath)
        return


def get_callbacks(callbacks=['save_model', 'early_stopping', 'log'], output_model_filepath=None, monitor='val_loss', model=None, data_collection=None, save_best_only=False, epoch_prediction_dir=None, batch_size=1, epoch_prediction_object=None, epoch_prediction_data_collection=None, epoch_prediction_batch_size=None, latent_size=128, backend='tensorflow', **kwargs):

    """ Very disorganized currently. Replace with dictionary? Also address never-ending parameters
    """

    print 'test error'
    return_callbacks = []

    for callback in callbacks:

        if callback == 'save_model':
            if backend == 'keras':
                return_callbacks += [ModelCheckpoint(output_model_filepath, monitor=monitor, save_best_only=save_best_only)]
            else:
                return_callbacks += [SaveModel(deepneuro_model=model)]

        if callback == 'early_stopping':
            return_callbacks += [EarlyStopping(monitor=monitor, patience=10)]

        if callback == 'log':
            return_callbacks += [CSVLogger(output_model_filepath.replace('.h5', '.log'))]

        if callback == 'predict_epoch':
            return_callbacks += [EpochPredict(deepneuro_model=model, data_collection=data_collection, epoch_prediction_dir=epoch_prediction_dir, batch_size=batch_size, epoch_prediction_object=epoch_prediction_object, epoch_prediction_data_collection=epoch_prediction_data_collection, epoch_prediction_batch_size=epoch_prediction_batch_size)]

        if callback == 'predict_gan':
            return_callbacks += [GANPredict(deepneuro_model=model, data_collection=data_collection, epoch_prediction_dir=epoch_prediction_dir, batch_size=batch_size, epoch_prediction_object=epoch_prediction_object, epoch_prediction_data_collection=epoch_prediction_data_collection, epoch_prediction_batch_size=epoch_prediction_batch_size, latent_size=latent_size)]

    return return_callbacks