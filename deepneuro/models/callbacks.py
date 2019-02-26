import os
import imageio
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from keras import backend as K

from deepneuro.utilities.util import add_parameter
from deepneuro.utilities.visualize import check_data


class CyclicLR(Callback):

    """

    The following callback class has been lifted from Brad Kenstler at https://github.com/bckenstler/CLR.
    Reproduced under MIT License.
    

    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in list(logs.items()):
            self.history.setdefault(k, []).append(v)
    
        K.set_value(self.model.optimizer.lr, self.clr())


class EpochPredict(Callback):

    """
    """

    def __init__(self, **kwargs):

        add_parameter(self, kwargs, 'epoch_prediction_data_collection', None)
        add_parameter(self, kwargs, 'epoch_prediction_object', None)
        add_parameter(self, kwargs, 'epoch_prediction_batch_size', 1)
        add_parameter(self, kwargs, 'epoch_prediction_dir', None)

        add_parameter(self, kwargs, 'deepneuro_model', None)
        
        add_parameter(self, kwargs, 'show_callback_output', False)
        add_parameter(self, kwargs, 'epoch_prediction_output_mode', 'gif')

        self.kwargs = kwargs

        if not os.path.exists(self.epoch_prediction_dir):
            os.mkdir(self.epoch_prediction_dir)

        self.predictions = []

        # Investigate why this doesn't happen by default.
        if self.epoch_prediction_batch_size is None:
            self.epoch_prediction_batch_size = 1

        # There's a more concise way to do this..
        self.predict_data = next(self.epoch_prediction_data_collection.data_generator(perpetual=True, verbose=False, just_one_batch=True, batch_size=self.epoch_prediction_batch_size))

    def on_train_end(self, logs={}):

        if self.predictions != []:

            if self.epoch_prediction_output_mode == 'gif':

                if type(self.predictions[0]) is list:
                    for output in range(len(self.predictions[0])):
                        current_predictions = [item[output] for item in self.predictions]
                        imageio.mimsave(os.path.join(self.epoch_prediction_dir, 'epoch_prediction_' + str(output) + '.gif'), current_predictions)
                else:
                    imageio.mimsave(os.path.join(self.epoch_prediction_dir, 'epoch_prediction.gif'), self.predictions)

            elif self.epoch_prediction_output_mode == 'mosaic':

                raise NotImplementedError('Training callback mosaics are not yet implemented. (epoch_prediction_output_mode = \'mosaic\'')

                if type(self.predictions[0]) is list:
                    for output in range(len(self.predictions[0])):
                        current_predictions = [item[output] for item in self.predictions]
                        prediction_array = np.array(current_predictions)
                        check_data({'Training Progress': prediction_array}, show_output=True, **self.kwargs)
                        print(prediction_array.shape)
                        # imageio.mimsave(os.path.join(self.epoch_prediction_dir, 'epoch_prediction_' + str(output) + '.gif'), current_predictions)
                else:
                    output_mosaic = np.array(self.predictions)
                    print(output_mosaic.shape)

            else:
                raise NotImplementedError  

        return
 
    def on_epoch_end(self, epoch, logs={}):

        if self.epoch_prediction_object is None:
            prediction = self.deepneuro_model.predict(self.predict_data[self.deepneuro_model.input_data])
        else:
            prediction = self.epoch_prediction_object.process_case(self.predict_data, model=self.deepneuro_model)

        output_filepaths, output_images = check_data({'prediction': prediction}, output_filepath=os.path.join(self.epoch_prediction_dir, 'epoch_{}.png'.format(epoch)), show_output=self.show_callback_output, batch_size=self.epoch_prediction_batch_size, **self.kwargs)

        if len(output_images.keys()) > 1:
            self.predictions += [[output_images['prediction_' + str(idx)].astype('uint8') for idx in range(len(output_images.keys()))]]
        else:
            self.predictions += [output_images['prediction'].astype('uint8')]

        return


class GANPredict(Callback):

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


class SaveModel(Callback):

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


def get_callbacks(callbacks=['save_model', 'early_stopping', 'log'], output_model_filepath=None, monitor='val_loss', model=None, data_collection=None, save_best_only=False, epoch_prediction_dir=None, batch_size=1, epoch_prediction_object=None, epoch_prediction_data_collection=None, epoch_prediction_batch_size=None, latent_size=128, backend='tensorflow', cyclic_base_learning_rate=.001, cyclic_max_learning_rate=.006, learning_rate_cycle=2000, **kwargs):

    """ Very disorganized currently. Replace with dictionary? Also address never-ending parameters
    """

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

        if callback == 'cyclic_learning_rate':
            return_callbacks += [CyclicLR(base_lr=cyclic_base_learning_rate, max_lr=cyclic_max_learning_rate, step_size=learning_rate_cycle)]

        if callback == 'predict_epoch':
            return_callbacks += [EpochPredict(deepneuro_model=model, prediction_data_collection=data_collection, epoch_prediction_dir=epoch_prediction_dir, epoch_prediction_object=epoch_prediction_object, epoch_prediction_data_collection=epoch_prediction_data_collection, epoch_prediction_batch_size=epoch_prediction_batch_size, **kwargs)]

        if callback == 'predict_gan':
            return_callbacks += [GANPredict(deepneuro_model=model, data_collection=data_collection, epoch_prediction_dir=epoch_prediction_dir, batch_size=batch_size, epoch_prediction_object=epoch_prediction_object, epoch_prediction_data_collection=epoch_prediction_data_collection, epoch_prediction_batch_size=epoch_prediction_batch_size, latent_size=latent_size)]

    return return_callbacks