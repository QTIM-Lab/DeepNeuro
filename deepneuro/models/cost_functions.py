from keras import backend as K


def cost_function_dict():

    return {'dice_coef': dice_coef,
            'dice_coef_loss': dice_coef_loss,
            }


def dice_coef_loss(y_true, y_pred):

    return (1 - dice_coef(y_true, y_pred))


def dice_coef(y_true, y_pred, smooth=1.):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
