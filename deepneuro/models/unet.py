from model import DeepNeuroModel, UpConvolution

from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, Nadam
from keras.layers.merge import concatenate

class UNet(DeepNeuroModel):

    def load(**kwargs):

        if 'depth' in kwargs:
            self.depth = kwargs.get('depth')
        else:
            self.depth = None

        if 'max_filter' in kwargs:
            self.max_filter = kwargs.get('max_filter')
        else:
            self.max_filter = 512

    def build_model():

        left_outputs = []

        for level in xrange(self.depth):

            filter_num = int(self.max_filter / (2 ^ (self.depth - level)) / self.downsize_filters_factor)

            if level == 0:
                left_outputs += [Conv3D(filter_num, self.filter_shape, activation=self.activation, padding='same')(inputs)]
                left_outputs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding='same')(left_outputs[level])
            else:
                left_outputs[level] = MaxPooling3D(pool_size=pool_size)(left_outputs[level-1])
                left_outputs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding='same')(left_outputs[level])
                left_outputs[level] = Conv3D(2 * filter_num, self.filter_shape, activation=self.activation, padding='same')(left_outputs[level])

            if self.dropout is not None:
                left_outputs[level] = Dropout(0.5)(left_outputs[level])

            if self.batch_norm:
                left_outputs[level] = BatchNormalization()(left_outputs[level])

        right_outputs = [left_outputs[self.depth - 1]]

        for level in xrange(self.depth):

            filter_num = int(self.max_filter / (2 ^ (level)) / self.downsize_filters_factor)

            if level > 0:
                right_outputs += [UpConvolution(pool_size=self.pool_size)(right_outputs[level-1])]
                right_outputs[level] = concatenate([right_outputs[level], left_outputs[self.depth - level - 1]], axis=4)
                right_outputs[level] = Conv3D(filter_num, self.filter_shape, activation=self.activation, padding='same')(right_outputs[level])
                right_outputs[level] = Conv3D(int(filter_num / 2), self.filter_shape, activation=self.activation, padding='same')(right_outputs[level])
            else:
                continue

            if self.dropout is not None:
                left_outputs[level] = Dropout(0.5)(right_outputs[level])

            if self.batch_norm:
                left_outputs[level] = BatchNormalization()(right_outputs[level])

        conv8 = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first')(conv7)

        pass

def u_net_3d(input_shape=None, input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.1, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True, activation='relu', output_shape=None):

    # Messy
    if input_tensor is not None:
        return conv8

    if regression:
        # act = Activation('relu')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq_loss])
    else:
        if num_outputs == 1:
            act = Activation('sigmoid')(conv8)
            model = Model(inputs=inputs, outputs=act)
            model.compile(optimizer=Nadam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
        else:
            act = Activation('softmax')(conv8)
            model = Model(inputs=inputs, outputs=act)
            model.compile(optimizer=Nadam(lr=initial_learning_rate), loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])

    return model