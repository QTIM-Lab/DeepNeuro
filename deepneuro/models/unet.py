""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

from keras.layers import Dropout, BatchNormalization, Lambda
from keras.layers.merge import concatenate

from deepneuro.models.keras_model import KerasModel
from deepneuro.models.dn_ops import DnConv, DnMaxPooling, DnDeConv, DnUpsampling, DnZeroPadding
from deepneuro.utilities.util import add_parameter


class UNet(KerasModel):
    
    def load(self, kwargs):

        """ Parameters
            ----------
            depth : int, optional
                Specified the layers deep the proposed U-Net should go.
                Layer depth is symmetric on both upsampling and downsampling
                arms.
            num_blocks : int, optional
                Number of consecutive convolutional layers at each resolution
                in the U-Net. Default is 2.

        """

        super(UNet, self).load(kwargs)

        add_parameter(self, kwargs, 'num_blocks', 2)
        add_parameter(self, kwargs, 'block_type', 'basic')
        add_parameter(self, kwargs, 'block_filter_growth_ratio', 2)
        add_parameter(self, kwargs, 'depth', 4)
        add_parameter(self, kwargs, 'output_channels', 1)
        add_parameter(self, kwargs, 'pooling_padding', self.padding)

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

                left_outputs += [DnConv(self.inputs, filter_num, kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_{}'.format(level, 0), backend='keras')]

                if self.dropout is not None and self.dropout != 0:
                    left_outputs[level] = Dropout(self.dropout)(left_outputs[level])

                if self.batch_norm:
                    left_outputs[level] = BatchNormalization()(left_outputs[level])

                for block_num in range(1, self.num_blocks):
                    left_outputs[level] = DnConv(left_outputs[level], filter_num * (self.block_filter_growth_ratio ** block_num), kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_{}'.format(level, block_num), backend='keras')

                    if self.dropout is not None and self.dropout != 0:
                        left_outputs[level] = Dropout(self.dropout)(left_outputs[level])

                    if self.batch_norm:
                        left_outputs[level] = BatchNormalization()(left_outputs[level])
            else:

                left_outputs += [DnMaxPooling(left_outputs[level - 1], pool_size=self.pool_size, dim=self.dim, backend='keras', padding=self.pooling_padding)]
                
                for block_num in range(self.num_blocks):
                    left_outputs[level] = DnConv(left_outputs[level], filter_num * (self.block_filter_growth_ratio ** block_num), kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='downsampling_conv_{}_{}'.format(level, block_num), backend='keras')

                    if self.dropout is not None and self.dropout != 0:
                        left_outputs[level] = Dropout(self.dropout)(left_outputs[level])

                    if self.batch_norm:
                        left_outputs[level] = BatchNormalization()(left_outputs[level])

        right_outputs = [left_outputs[self.depth - 1]]

        for level in range(self.depth):

            filter_num = int(self.max_filter / (2 ** (level)) / self.downsize_filters_factor)

            if level > 0:
                right_outputs += [DnUpsampling(right_outputs[level - 1], pool_size=self.pool_size, dim=self.dim, backend='keras')]

                if right_outputs[level].shape[1:-1] == left_outputs[self.depth - level - 1].shape[1:-1]:
                    right_outputs[level] = concatenate([right_outputs[level], left_outputs[self.depth - level - 1]], axis=self.dim + 1)
                else:
                    # Very complex code to facilitate dimension errors arising from odd-numbered patches.
                    # Is essentially same-padding, may have performance impacts on networks.
                    # Very tentative.
                    input_tensor = right_outputs[level]
                    concatenate_tensor = left_outputs[self.depth - level - 1]
                    padding = []
                    for dim in range(self.dim):
                        padding += [int(concatenate_tensor.shape[dim + 1]) - int(input_tensor.shape[dim + 1])]
                    if len(padding) < self.dim:
                        padding += [1] * self.dim - len(self.padding)
                    lambda_dict = {0: Lambda(lambda x: x[:, 0:padding[0], :, :, :]),
                                    1: Lambda(lambda x: x[:, :, 0:padding[1], :, :]),
                                    2: Lambda(lambda x: x[:, :, :, 0:padding[2], :])}
                    for dim in range(self.dim):
                        tensor_slice = [slice(None)] + [slice(padding[dim]) if i_dim == dim else slice(None) for i_dim in range(self.dim)] + [slice(None)]

                        # Causes JSON Serialization Error.
                        # tensor_slice = Lambda(lambda x, tensor_slice=tensor_slice: x[tensor_slice])(input_tensor)

                        tensor_slice = lambda_dict[dim](input_tensor)
                        input_tensor = concatenate([input_tensor, tensor_slice], axis=dim + 1)
                    right_outputs[level] = concatenate([input_tensor, concatenate_tensor], axis=self.dim + 1)

                for block_num in range(self.num_blocks):
                    right_outputs[level] = DnConv(right_outputs[level], filter_num // (self.block_filter_growth_ratio ** block_num), kernel_size=self.kernel_size, stride_size=(1,) * self.dim, activation=self.activation, padding=self.padding, dim=self.dim, name='upsampling_conv_{}_{}'.format(level, block_num), backend='keras')

                    if self.dropout is not None and self.dropout != 0:
                        right_outputs[level] = Dropout(self.dropout)(right_outputs[level])

                    if self.batch_norm:
                        right_outputs[level] = BatchNormalization()(right_outputs[level])

        self.output_layer = DnConv(right_outputs[level], self.output_channels, (1, ) * self.dim, stride_size=(1,) * self.dim, dim=self.dim, name='end_conv', backend='keras') 

        super(UNet, self).build_model()
