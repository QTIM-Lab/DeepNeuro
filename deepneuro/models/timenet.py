""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

import numpy as np

from model import DeepNeuroModel, UpConvolution

from keras.engine import Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Dropout, BatchNormalization, TimeDistributed, Reshape, Dense, LSTM
from keras.optimizers import Nadam
from keras.layers.merge import concatenate

from cost_functions import dice_coef_loss, dice_coef


class TimeNet(DeepNeuroModel):
    
    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            max_filter: int, optional
                Specifies the number of filters at the bottom level of the U-Net.

        """

        if 'depth' in kwargs:
            self.depth = kwargs.get('depth')
        else:
            self.depth = 4

        if 'max_filter' in kwargs:
            self.max_filter = kwargs.get('max_filter')
        else:
            self.max_filter = 128

    def build_model(self):
        
        """ A basic implementation of the U-Net proposed in https://arxiv.org/abs/1505.04597
        
            TODO: specify optimizer

            Returns
            -------
            Keras model or tensor
                If input_tensor is provided, this will return a tensor. Otherwise,
                this will return a Keras model.
        """

        # downsample_arm = []
        # middle_arm = []
        # upsample_arm = []

        # for level in xrange(self.depth):

        #     filter_num = max_filter

        #     if level == 0:
        #         downsample_arm += [TimeDistributed(Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding))(self.inputs)]
        #     else:
        #         downsample_arm += [TimeDistributed(MaxPooling3D(pool_size=self.pool_size))(downsample_arm[level - 1])]
        #         downsample_arm[level] = TimeDistributed(Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding))(downsample_arm[level])

        #     if self.dropout is not None:
        #         downsample_arm[level] = Dropout(self.dropout)(downsample_arm[level])

        #     if self.batch_norm:
        #         downsample_arm[level] = BatchNormalization()(downsample_arm[level])

        # cells = []

        # for level in xrange(self.depth):

        #     filter_num = int(self.max_filter / (2 ** (level)) / self.downsize_filters_factor)
        #     print filter_num

        #     if level == 0:
        #         cells += [TimeDistributed(Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding))(self.inputs)]
        #     else:
        #         cells += [TimeDistributed(Conv3D(filter_num, self.filter_shape, activation=self.activation, padding=self.padding))(cells[-1])]

        #     if self.dropout is not None:
        #         cells[level] = TimeDistributed(Dropout(self.dropout))(cells[level])

        #     if self.batch_norm:
        #         cells[level] = TimeDistributed(BatchNormalization())(cells[level])

        # layer_1 = TimeDistributed(Conv3D(32, self.filter_shape, activation=self.activation, padding=self.padding))(self.inputs)
        # layer_2 = TimeDistributed(Conv3D(32, self.filter_shape, activation=self.activation, padding=self.padding))(layer_1)
        # layer_3 = TimeDistributed(Conv3D(1, self.filter_shape, activation=self.activation, padding=self.padding))(layer_1)

        # feature_layer = Reshape((self.input_shape[1], self.input_shape[2], self.input_shape[3], 64*self.input_shape[0]))(layer_1)

        print self.input_shape
        rnn_feature_num = self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
        dense_layer = Reshape((self.input_shape[0], -1))(self.inputs)
        rnn_layer = LSTM(int(rnn_feature_num*32), return_sequences=True, activation=self.activation)(dense_layer)
        rnn_layer = LSTM(int(rnn_feature_num*16), return_sequences=True, activation=self.activation)(rnn_layer)
        rnn_layer = LSTM(int(rnn_feature_num*8), return_sequences=True, activation=self.activation)(rnn_layer)
        rnn_layer = LSTM(int(rnn_feature_num*4), return_sequences=True, activation=self.activation)(rnn_layer)
        rnn_layer = LSTM(int(rnn_feature_num*2), return_sequences=True, activation=self.activation)(rnn_layer)
        rnn_layer = LSTM(int(rnn_feature_num), return_sequences=False)(rnn_layer)
        reformed_layer = Reshape((self.input_shape[1], self.input_shape[2], self.input_shape[3], 1))(rnn_layer)
        
        # output_layer_3 = Conv3D(32, self.pool_size, activation=self.activation, padding=self.padding)(reformed_layer)
        # output_layer_2 = Conv3D(32, self.filter_shape, activation=self.activation, padding=self.padding)(reformed_layer)
        # output_layer_1 = Conv3D(int(self.num_outputs), (1, 1, 1))(output_layer_2)
        output_layer = reformed_layer

        # TODO: Brainstorm better way to specify outputs
        if self.input_tensor is not None:
            return output_layer

        if self.output_type == 'regression':
            self.model = Model(inputs=self.inputs, outputs=output_layer)
            self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

        if self.output_type == 'binary_label' or self.num_outputs > 1:
            act = Activation('sigmoid')(output_layer)
            self.model = Model(inputs=self.inputs, outputs=act)
            self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

        if self.output_type == 'categorical_label' or self.num_outputs > 1:
            act = Activation('softmax')(output_layer)
            self.model = Model(inputs=self.inputs, outputs=act)
            self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])

        return self.model