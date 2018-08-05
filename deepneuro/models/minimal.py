""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

from keras.engine import Model
from keras.layers import Conv3D, MaxPooling3D, Activation, Dropout, BatchNormalization
from keras.optimizers import Nadam
from keras.layers.merge import concatenate

from deepneuro.models.model import KerasModel
from deepneuro.models.cost_functions import dice_coef_loss, dice_coef
from deepneuro.models.dn_ops import DnConv, DnMaxPooling, DnDeConv, DnUpsampling
from deepneuro.utilities.util import add_parameter


class MinimalKerasCNN(KerasModel):
    
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

        super(MinimalKerasCNN, self).load(kwargs)

        add_parameter(self, kwargs, 'filter_size', 1)

    def build_model(self):

        output_layer = DnConv(self.inputs, 1, self.kernel_size, stride_size=(1,) * self.dim, dim=self.dim, name='minimal_conv', backend='keras') 

        # TODO: Brainstorm better way to specify outputs
        if self.input_tensor is not None:
            return output_layer

        if self.output_type == 'regression':
            self.model = Model(inputs=self.inputs, outputs=output_layer)
            self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

        if self.output_type == 'binary_label':
            act = Activation('sigmoid')(output_layer)
            self.model = Model(inputs=self.inputs, outputs=act)
            self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

        if self.output_type == 'categorical_label':
            act = Activation('softmax')(output_layer)
            self.model = Model(inputs=self.inputs, outputs=act)
            self.model.compile(optimizer=Nadam(lr=self.initial_learning_rate), loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])

        super(MinimalKerasCNN, self).build()

        return self.model