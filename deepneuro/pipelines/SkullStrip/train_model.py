
from deepneuro.models.unet import UNet
from deepneuro.models.train import train_model


def train_model():

    model = UNet()

    train_model(model, 'SkullStrip_TestModel.h5')

    print model


def load_data():

    return


if __name__ == '__main__':
    train_model()