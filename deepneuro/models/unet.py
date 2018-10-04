""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

from keras.layers import Dropout, BatchNormalization
from keras.layers.merge import concatenate

from deepneuro.models.keras_model import KerasModel
from deepneuro.models.dn_ops import DnConv, DnMaxPooling, DnDeConv, DnUpsampling
from deepneuro.utilities.util import add_parameter


class UNet(KerasModel):
    
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

        super(UNet, self).load(kwargs)

        add_parameter(self, kwargs, 'depth', 4)
        add_parameter(self, kwargs, 'output_channels', 1)

    def build_model(self):
        
        """ A basic implementation of the U-Net proposed in https://arxiv.org/abs/1505.04597
        
            TODO: specify optimizer

            Returns
            -------
            Keras model or tensor
                If input_tensor is provided, this will return a tensor. Otherwise,
                this will return a Keras model.
        """

        left_outputs = []

        for level in range(self.depth):

            filter_num = int(self.max_filter / (2 ** (self.depth - level)) / self.downsize_filters_factor)

            if level == 0:
                left_outputs += [DnConv(self.inputs, filter_num, kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_1'.format(level), backend='keras')]
                left_outputs[level] = DnConv(left_outputs[level], 2 * filter_num, kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_2'.format(level), backend='keras')
            else:
                left_outputs += [DnMaxPooling(left_outputs[level - 1], pool_size=self.pool_size, dim=self.dim, backend='keras')]
                left_outputs[level] = DnConv(left_outputs[level], filter_num, kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_1'.format(level), backend='keras')
                left_outputs[level] = DnConv(left_outputs[level], 2 * filter_num, kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_2'.format(level), backend='keras')

            if self.dropout is not None and self.dropout != 0:
                left_outputs[level] = Dropout(self.dropout)(left_outputs[level])

            if self.batch_norm:
                left_outputs[level] = BatchNormalization()(left_outputs[level])

        right_outputs = [left_outputs[self.depth - 1]]

        for level in range(self.depth):

            filter_num = int(self.max_filter / (2 ** (level)) / self.downsize_filters_factor)

            if level > 0:
                right_outputs += [DnUpsampling(right_outputs[level - 1], pool_size=self.pool_size, dim=self.dim, backend='keras')]
                right_outputs[level] = concatenate([right_outputs[level], left_outputs[self.depth - level - 1]], axis=self.dim + 1)
                right_outputs[level] = DnConv(right_outputs[level], filter_num, kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='upsampling_conv_{}_1'.format(level), backend='keras')
                right_outputs[level] = DnConv(right_outputs[level], int(filter_num / 2), kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='upsampling_conv_{}_2'.format(level), backend='keras')
            else:
                continue

            if self.dropout is not None and self.dropout != 0:
                right_outputs[level] = Dropout(self.dropout)(right_outputs[level])

            if self.batch_norm:
                right_outputs[level] = BatchNormalization()(right_outputs[level])

        self.output_layer = DnConv(right_outputs[level], self.output_channels, (1, ) * self.dim, stride_size=(1,) * self.dim, dim=self.dim, name='end_conv', backend='keras') 

        super(UNet, self).build_model()
