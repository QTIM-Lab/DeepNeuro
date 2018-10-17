from keras.engine import Input, Model
from keras.layers import Activation
from keras.optimizers import Nadam, SGD, Adam
from keras import backend as K

from deepneuro.models.cost_functions import dice_coef_loss, dice_coef
from deepneuro.models.model import DeepNeuroModel
from deepneuro.utilities.visualize import check_data
from deepneuro.utilities.util import add_parameter
from deepneuro.models.callbacks import get_callbacks


class KerasModel(DeepNeuroModel):

    def load(self, kwargs):

        super(KerasModel, self).load(kwargs)

        add_parameter(self, kwargs, 'output_activation', True)

        self.keras_optimizer_dict = {'Nadam': Nadam, 'Adam': Adam, 'SGD': SGD}

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

        # Todo: investigate call-backs more thoroughly.
        # Also, maybe something more general for the difference between training and validation.
        # Todo: list-checking for callbacks

        self.create_data_generators(training_data_collection, validation_data_collection, input_groups, training_batch_size, validation_batch_size, training_steps_per_epoch, validation_steps_per_epoch)

        self.callbacks = get_callbacks(callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, model=self, batch_size=training_batch_size, backend='keras', **kwargs)

        try:
            if validation_data_collection is None:
                self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, callbacks=self.callbacks)
            else:
                self.model.fit_generator(generator=self.training_data_generator, steps_per_epoch=self.training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, validation_data=self.validation_data_generator, validation_steps=self.validation_steps_per_epoch, callbacks=self.callbacks)
        except KeyboardInterrupt:
            for callback in self.callbacks:
                callback.on_train_end()
        except:
            raise

        return

    def fit_one_batch(self, training_data_collection, output_model_filepath=None, input_groups=None, output_directory=None, callbacks=['save_model', 'log'], training_batch_size=16, training_steps_per_epoch=None, num_epochs=None, show_results=True, **kwargs):

        one_batch_generator = self.keras_generator(training_data_collection.data_generator(perpetual=True, data_group_labels=input_groups, verbose=False, just_one_batch=True, batch_size=training_batch_size))

        self.callbacks = get_callbacks(callbacks, output_model_filepath=output_model_filepath, data_collection=training_data_collection, model=self, batch_size=training_batch_size, backend='keras', **kwargs)

        if training_steps_per_epoch is None:
            training_steps_per_epoch = training_data_collection.total_cases // training_batch_size + 1

        try:
            self.model.fit_generator(generator=one_batch_generator, steps_per_epoch=training_steps_per_epoch, epochs=num_epochs, pickle_safe=True, callbacks=self.callbacks)
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

    def build_model(self):

        # TODO: Brainstorm better way to specify outputs
        if self.input_tensor is None:

            if self.output_type == 'msq':
                self.model = Model(inputs=self.inputs, outputs=self.output_layer)
                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

            if self.output_type == 'dice':
                if self.output_activation:
                    self.model = Model(inputs=self.inputs, outputs=Activation('sigmoid')(self.output_layer))
                else:
                    self.model = Model(inputs=self.inputs, outputs=self.output_layer)
                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

            if self.output_type == 'multi_dice':
                if self.output_activation:
                    self.model = Model(inputs=self.inputs, outputs=Activation('sigmoid')(self.output_layer))
                else:
                    self.model = Model(inputs=self.inputs, outputs=self.output_layer)
                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

            if self.output_type == 'binary_crossentropy':
                if self.output_activation:
                    self.model = Model(inputs=self.inputs, outputs=Activation('sigmoid')(self.output_layer))
                else:
                    self.model = Model(inputs=self.inputs, outputs=self.output_layer)
                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])

            if self.output_type == 'categorical_crossentropy':
                if self.output_activation:
                    self.model = Model(inputs=self.inputs, outputs=Activation('softmax')(self.output_layer))
                else:
                    self.model = Model(inputs=self.inputs, outputs=self.output_layer)
                self.model.compile(optimizer=self.keras_optimizer_dict[self.optimizer](lr=self.initial_learning_rate), loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])

            self.model_input_shape = self.model.layers[0].input_shape
            self.model_output_shape = self.model.layers[-1].output_shape

            return self.model

        else:

            return self.output_layer


if __name__ == '__main__':

    pass
