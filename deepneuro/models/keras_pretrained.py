""" Pretrained neural networks implemented by Keras.
    Implementation borrows heavily from: 
    https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
    https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
    written by Adrian Rosebrock and Felix Yu, respectively
"""

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.engine import Model

from deepneuro.models.keras_model import KerasModel
from deepneuro.utilities.util import add_parameter


class KerasPreTrainedModel(KerasModel):
    
    def load(self, kwargs):

        """ Parameters
            ----------
            model_version : str, optional
                

        """

        super(KerasPreTrainedModel, self).load(kwargs)

        # Model Choice Parameters
        add_parameter(self, kwargs, 'model_type', 'inception')
        add_parameter(self, kwargs, 'pretrained_weights', 'imagenet')

        # Finetuning Parameters
        add_parameter(self, kwargs, 'input_shape', None)
        add_parameter(self, kwargs, 'output_classes', None)
        add_parameter(self, kwargs, 'bottleneck_layers_num', None)
        add_parameter(self, kwargs, 'additional_dense_layers', 0)
        add_parameter(self, kwargs, 'additional_dense_feature_num', 128)

        self.models = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "vgg": VGG16,
            "inception": InceptionV3,
            "xception": Xception,
            "resnet50": ResNet50,
            "resnet": ResNet50,
            "inception": InceptionV3,
            "densenet": DenseNet121
        }

        self.output_activation = True

        if self.input_shape is None:
            self.include_top = False
        else:
            self.include_top = True

    def build_model(self):
        
        self.model = self.models[self.model_type](weights=self.pretrained_weights, include_top=False)

        if self.output_classes is not None:

            # This may need to be made architecture-specific.
            model_output = self.model.output
            model_output = GlobalAveragePooling2D()(model_output)

            for i in range(self.additional_dense_layers):
                model_output = Dense(self.additional_dense_feature_num, activation='relu')(model_output)

            predictions = Dense(self.output_classes)(model_output)

        if self.bottleneck_layers_num is not None:
            for layer in self.model.layers[:self.bottleneck_layers_num]:
                layer.trainable = False

        self.output_layer = predictions
        self.inputs = self.model.input

        super(KerasPreTrainedModel, self).build_model(compute_output=True)
