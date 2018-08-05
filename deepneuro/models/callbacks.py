import keras
import os
import imageio

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from deepneuro.utilities.util import add_parameter
from deepneuro.utilities.visualize import check_data


class EpochPredict(keras.callbacks.Callback):

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'data_collection', None)
        add_parameter(self, kwargs, 'epoch_prediction_data_collection', self.data_collection)
        add_parameter(self, kwargs, 'epoch_prediction_object', None)
        add_parameter(self, kwargs, 'deepneuro_model', None)
        add_parameter(self, kwargs, 'output_folder', None)
        add_parameter(self, kwargs, 'output_gif', None)
        add_parameter(self, kwargs, 'batch_size', 1)
        add_parameter(self, kwargs, 'epoch_prediction_batch_size', self.batch_size)

        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        if self.epoch_prediction_object is not None:
            self.epoch_prediction_object.model = self.deepneuro_model
        else:
            self.epoch_prediction_object = self.deepneuro_model

        self.predictions = []

        # There's a more concise way to do this..
        self.predict_data = next(self.epoch_prediction_data_collection.data_generator(perpetual=True, verbose=False, just_one_batch=True, batch_size=self.epoch_prediction_batch_size))

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        imageio.mimsave(os.path.join(self.output_folder, 'epoch_prediction.gif'), self.predictions)
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):

        prediction = self.epoch_prediction_object.process_case(self.predict_data[self.deepneuro_model.input_data], model=self.deepneuro_model)

        output_filepaths, output_images = check_data({'prediction': prediction}, output_filepath=os.path.join(self.output_folder, 'epoch_{}.png'.format(epoch)), show_output=False, batch_size=self.epoch_prediction_batch_size)

        self.predictions += [output_images['prediction'].astype('uint8')]

        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


def get_callbacks(model_file, callbacks=['save_model', 'early_stopping', 'log'], monitor='val_loss', model=None, data_collection=None, save_best_only=False, epoch_prediction_dir=None, batch_size=1, epoch_prediction_object=None, epoch_prediction_data_collection=None, epoch_prediction_batch_size=None):

    """ Temporary function; callbacks will be dealt with in more detail in the future.
        Very disorganized currently. Do with dictionary. 
    """

    return_callbacks = []
    for callback in callbacks:
        if callback == 'save_model':
            return_callbacks += [ModelCheckpoint(model_file, monitor=monitor, save_best_only=save_best_only)]
        if callback == 'early_stopping':
            return_callbacks += [EarlyStopping(monitor=monitor, patience=10)]
        if callback == 'log':
            return_callbacks += [CSVLogger(model_file.replace('.h5', '.log'))]
        if callback == 'predict_epoch':
            return_callbacks += [EpochPredict(deepneuro_model=model, data_collection=data_collection, output_folder=epoch_prediction_dir, batch_size=batch_size, epoch_prediction_object=epoch_prediction_object, epoch_prediction_data_collection=epoch_prediction_data_collection, epoch_prediction_batch_size=epoch_prediction_batch_size)]
    return return_callbacks