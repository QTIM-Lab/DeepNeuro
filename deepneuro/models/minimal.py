""" unet.py includes different implementations of the popular U-Net model.
    See more at https://arxiv.org/abs/1505.04597
"""

from deepneuro.models.keras_model import KerasModel
from deepneuro.models.dn_ops import DnConv


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

    def build_model(self):

        self.output_layer = DnConv(self.inputs, 1, self.kernel_size, stride_size=(1,) * self.dim, dim=self.dim, name='minimal_conv', backend='keras') 

        if self.input_tensor is None:

            super(MinimalKerasCNN, self).build()
            return self.model

        else:
            return self.output_layer