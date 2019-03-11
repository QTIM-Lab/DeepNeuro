from keras.engine import Input, Model
from keras.layers import Activation, Lambda
from keras.layers.merge import concatenate
from keras.optimizers import Nadam, SGD, Adam, RMSprop, Adagrad, Adamax, Adadelta
from keras import backend as K

from deepneuro.models.cost_functions import dice_coef_loss, dice_coef
from deepneuro.models.model import DeepNeuroModel
from deepneuro.utilities.visualize import check_data
from deepneuro.utilities.util import add_parameter
from deepneuro.models.callbacks import get_callbacks
from deepneuro.models.cost_functions import WeightedCategoricalCrossEntropy


class KerasModel(DeepNeuroModel):

    def load(self, kwargs):

        super(KerasModel, self).load(kwargs)

        # Basic Keras Model Params
        add_parameter(self, kwargs, 'output_activation', True)
        add_parameter(self, kwargs, 'class_weights', True)

        # Specific Cost Function Params
        add_parameter(self, kwargs, 'categorical_weighting', {0: 0.1, 1: 3.0})

        self.keras_optimizer_dict = {'Nadam': Nadam, 'Adam': Adam, 'SGD': SGD, 'RMSprop': RMSprop, 'Adagrad': Adagrad, 'Adamax': Adamax, 'Adadelta': Adadelta}

        if self.input_tensor is None:
            self.inputs = Input(self.input_shape)
        else:
            self.inputs = self.input_tensor

    def train(self, training_data_collection, validation_data_collection=None, output_model_filepath=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None, initial_learning_rate=.0001, learning_rate_drop=None, learning_rate_epochs=None, num_epochs=None, callbacks=['save_model', 'log'], **kwargs):

        """
        input_groups : list of strings, optional
            Specifies which named data groups (e.g. "ground_truth") enter which input
            data slot in your model.
        """

        self.create_data_generators(training_data_collection, validation_data_collection, input_groups, training_batch_size, validation_batch_size, training_steps_per_epoch, validation_steps_per_epoch)

        self.callbacks = get_callbacks(callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, model=self, batch_size=training_batch_size, backend='keras', **kwargs)

        try:
            if validation_data_collection is None:
                self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, callbacks=self.callbacks)
            else:
                self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, validation_data=self.validation_data_generator, validation_steps=self.validation_steps_per_epoch, callbacks=self.callbacks, workers=0)
        except KeyboardInterrupt:
            for callback in self.callbacks:
                callback.on_train_end()
        except:
            raise

        return

    def fit_one_batch(self, training_data_collection, output_model_filepath=None, input_groups=None, output_directory=None, callbacks=['save_model', 'log'], training_batch_size=16, training_steps_per_epoch=None, num_epochs=None, show_results=False, **kwargs):

        one_batch_generator = self.keras_generator(training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, just_one_batch=True, batch_size=training_batch_size))

        self.callbacks = get_callbacks(callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, model=self, batch_size=training_batch_size, backend='keras', **kwargs)

        if training_steps_per_epoch is None:
            training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1

        try:
            self.model.fit_generator(generator=one_batch_generator, 
                steps_per_epoch=training_steps_per_epoch, 
                epochs=num_epochs,
                callbacks=self.callbacks)
        except KeyboardInterrupt:
            for callback in self.callbacks:
                callback.on_train_end()
        except:
            raise

        one_batch = next(one_batch_generator)
        prediction = self.predict(one_batch[0])

        if show_results:
            check_data(output_data={self.input_data: one_batch[0], self.targets: one_batch[1], 'prediction': prediction}, batch_size=training_batch_size)

        return

    def create_data_generators(self, training_data_collection, validation_data_collection=None, input_groups=None, training_batch_size=32, validation_batch_size=32, training_steps_per_epoch=None, validation_steps_per_epoch=None):

        super(KerasModel, self).create_data_generators(training_data_collection, validation_data_collection, input_groups, training_batch_size, validation_batch_size, training_steps_per_epoch, validation_steps_per_epoch)

        self.training_data_generator = self.keras_generator(self.training_data_generator)

        if self.validation_data_generator is not None:
            self.validation_data_generator = self.keras_generator(self.validation_data_generator)

        return

    def keras_generator(self, data_generator, input_data='input_data', targets='ground_truth'):

        while True:
            data = next(data_generator)
            keras_data = (data[self.input_data], data[self.targets])
            yield keras_data

        return

    def get_layer_output_shape(self, layer_num):

        return self.model.layers[layer_num].output_shape

    def get_layer_output_function(self, layer_num):

        """ Only works for the sequential model.
        """

        return K.function([self.model.layers[0].input], [self.model.layers[layer_num].output])

    def predict(self, input_data):

        return self.model.predict(input_data)

    def build_model(self, compute_output=True):

        """ 
        """

        # TODO: Move this entire section to cost_functions.py

        if self.input_tensor is None:

            if self.cost_function == 'mse':

                if compute_output:
                    self.model = Model(inputs=self.inputs, outputs=self.output_layer)

                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

            elif self.cost_function == 'dice':

                if compute_output:
                    if self.output_activation:
                        self.model = Model(inputs=self.inputs, outputs=Activation('sigmoid')(self.output_layer))
                    else:
                        self.model = Model(inputs=self.inputs, outputs=self.output_layer)

                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

            # Not Implemented
            elif self.cost_function == 'multi_dice':

                raise NotImplementedError('Multi-dice coefficient not yet implemented.')
                
                if compute_output:
                    if self.output_activation:
                        self.model = Model(inputs=self.inputs, outputs=Activation('sigmoid')(self.output_layer))
                    else:
                        self.model = Model(inputs=self.inputs, outputs=self.output_layer)

                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

            elif self.cost_function == 'binary_crossentropy':

                if compute_output:
                    if self.output_activation:
                        self.model = Model(inputs=self.inputs, outputs=Activation('sigmoid')(self.output_layer))
                    else:
                        self.model = Model(inputs=self.inputs, outputs=self.output_layer)

                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])

            elif self.cost_function == 'categorical_crossentropy':

                if compute_output:
                    if self.output_activation:
                        self.model = Model(inputs=self.inputs, outputs=Activation('softmax')(self.output_layer))
                    else:
                        self.model = Model(inputs=self.inputs, outputs=self.output_layer)

                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            elif self.cost_function == 'weighted_categorical_crossentropy':

                if compute_output:
                    activation = Activation('sigmoid')(self.output_layer)
                    activation_categorical = Lambda(lambda arg: K.ones_like(arg) - arg)(activation)
                    predictions = concatenate([activation, activation_categorical], axis=-1)

                    if self.output_activation:
                        self.model = Model(inputs=self.inputs, outputs=predictions)
                    else:
                        self.model = Model(inputs=self.inputs, outputs=self.output_layer)

                lossFunc = WeightedCategoricalCrossEntropy(self.categorical_weighting)
                self.model.compile(self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss=lossFunc.loss_wcc_dist, metrics=[lossFunc.metric_dice_dist, lossFunc.metric_acc])

            else:
                raise NotImplementedError('Cost function {} not implemented.'.format(self.cost_function))

            self.model_input_shape = self.model.layers[0].input_shape
            self.model_output_shape = self.model.layers[-1].output_shape

            return self.model

        else:

            self.model_input_shape = self.model.layers[0].input_shape
            self.model_output_shape = self.model.layers[-1].output_shape

            return self.output_layer

    def load_weights(self, weights_file):

        self.model.load_weights(weights_file)

        return


if __name__ == '__main__':

    pass
