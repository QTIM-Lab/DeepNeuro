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
        add_parameter(self, kwargs, 'finetuning_dense_features', 128)

        self.models = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception": Xception,
            "resnet50": ResNet50
        }

        self.output_activation = False

        if self.input_shape is None:
            self.include_top = False
        else:
            self.include_top = True

    def build_model(self):
        
        self.model = self.models[self.model_type](weights=self.pretrained_weights, include_top=False)

        if self.output_classes is not None:

            if self.model_type in ['vgg19', 'vgg16']:
                self.model.layers.pop()
                self.model.outputs = [self.model.layers[-1].output]
                self.model.layers[-1].outbound_nodes = []
                self.model.add(Dense(self.output_classes, activation='softmax'))

            elif self.model_type in ['inception']:
                model_output = self.model.output
                model_output = GlobalAveragePooling2D()(model_output)
                model_output = Dense(self.finetuning_dense_features, activation='relu')(model_output)
                predictions = Dense(self.output_classes, activation='softmax')(model_output)
                self.model = Model(self.model.input, predictions)
            elif self.model_type in ['resnet50']:
                # self.model.layers.pop()
                model_output = self.model.output
                model_output = GlobalAveragePooling2D()(model_output)
                predictions = Dense(self.output_classes, activation='softmax')(model_output)
                self.model = Model(self.model.input, predictions)

        if self.bottleneck_layers_num is not None:
            for layer in self.model.layers[:self.bottleneck_layers_num]:
                layer.trainable = False

        self.inputs = self.model.input

        super(KerasPreTrainedModel, self).build_model(compute_output=False)
