import numpy as np
import itertools


def cost_function_dict(wcc_weights={0: 0.1, 1: 3.0}, **kwargs):
    
    """Summary
    
    Parameters
    ----------
    **kwargs
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    
    lossFunc = WeightedCategoricalCrossEntropy(wcc_weights)

    return {'dice_coef': dice_coef,
            'dice_coef_loss': dice_coef_loss,
            'multi_dice_coef': multi_dice_coef,
            'multi_dice_loss': multi_dice_loss,
            'loss_wcc': lossFunc.loss_wcc,
            'metric_wcc': lossFunc.metric_wcc,
            'loss_wcc_dist': lossFunc.loss_wcc_dist, 
            'loss_dice': lossFunc.loss_dice,
            'metric_dice': lossFunc.metric_dice, 
            'metric_dice_dist': lossFunc.metric_dice_dist, 
            'metric_acc': lossFunc.metric_acc
            }


def dice_coef(y_true, y_pred, smooth=1.):
    """Summary
    
    Parameters
    ----------
    y_true : Tensor
        Ground truth values for input to a cost function.
    y_pred : Tensor
        Predicted values for input to a cost function.
    smooth : float, optional
        Description
    
    Returns
    -------
    Tensor
        Soft-dice coefficient ranging from 0 to 1.
    """
    from keras import backend as K

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    
    """ A version of the soft-dice coefficient loss implemented in deepneuro.models.cost_functions.dice_coef
        that can be minimzed while training a model.
    
    Parameters
    ----------
    y_true : Tensor
        Ground truth values for input to a cost function.
    y_pred : Tensor
        Predicted values for input to a cost function.
    
    Returns
    -------
    Tensor
        Soft-dice loss ranging from 0 to 1.
    """
    
    return (1 - dice_coef(y_true, y_pred))


def multi_dice_coef(y_true, y_pred, channels):

    def dice_coef(y_true, y_pred):

        dice_list = []
        for channel in range(channels):
            dice_list += [dice_coef(y_true[..., channel], y_pred[..., channel])]

        from keras.layers import Average

        return Average(dice_list)

    return dice_coef


def multi_dice_loss(y_true, y_pred, channels):

    def dice_loss(y_true, y_pred):
        return (1 - multi_dice_coef(y_true, y_pred, channels))

    return dice_loss


def wasserstein_loss(model, discriminator, discriminator_fake_logits, discriminator_real_logits, synthetic_images, reference_images, gradient_penalty_weight=10, name='discriminator', dim=2, depth=None, transition=False, alpha_transition=0):

    import tensorflow as tf

    if depth is None:
        depth = model.depth

    D_loss = tf.reduce_mean(discriminator_fake_logits) - tf.reduce_mean(discriminator_real_logits)
    G_loss = -tf.reduce_mean(discriminator_fake_logits)

    differences = synthetic_images - reference_images
    alpha = tf.random_uniform(shape=[tf.shape(differences)[0]] + [1] * (dim + 1), minval=0., maxval=1.)
    interpolates = reference_images + (alpha * differences)
    _, interpolates_logits = discriminator(model, interpolates, reuse=True, depth=depth, name=name, transition=transition, alpha_transition=alpha_transition)
    gradients = tf.gradients(interpolates_logits, [interpolates])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=list(range(1, 2 + model.dim))))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    tf.summary.scalar("gp_loss", gradient_penalty)

    D_origin_loss = D_loss
    D_loss += 10 * gradient_penalty
    D_loss += 0.001 * tf.reduce_mean(tf.square(discriminator_real_logits - 0.0))

    return [D_loss], [G_loss], [D_origin_loss]


def focal_loss(y_true, y_pred, gamma=2):

    from keras import backend as K

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    return -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred), axis=-1)


class WeightedCategoricalCrossEntropy(object):

    """ This code was developed by jbpatel@mit.edu.
    """

    def __init__(self, weights):

        nb_cl = len(weights)
        self.weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'w_categorical_crossentropy_function'
        self.metric = 0.0
        self.diceMetric = 0.0
        self.smooth = 1.0

    def loss_wcc(self, y_true, y_pred):
        return self.w_categorical_crossentropy_function(y_true, y_pred)

    def loss_wcc_dist(self, y_true, y_pred):

        from keras import backend as K
        from keras.layers import concatenate, Lambda

        y_dist = Lambda((lambda arg: K.expand_dims(arg[..., 1], axis=-1)))(y_true)
        y_true1 = Lambda((lambda arg: K.expand_dims(arg[..., 0], axis=-1)))(y_true)
        y_true_complement = Lambda(lambda arg: (K.ones_like(arg) - arg))(y_true1)
        y_true_onehot = concatenate([y_true_complement, y_true1], axis=-1)
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        # y_pred_max_mat = K.equal(y_pred, y_pred_max)

        wcc_loss = self.w_categorical_crossentropy_function(y_true1, y_pred)
        distTrans_loss = self.distTransErrorMatrix(y_true_onehot, y_dist, y_pred)
        total_loss = wcc_loss * distTrans_loss

        return total_loss

    def metric_wcc(self, y_true, y_pred):
        return self.metric

    def w_categorical_crossentropy_function(self, y_true, y_pred):
        
        from keras import backend as K
        from keras.layers import concatenate, Lambda

        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)

        #change y_true into one-hot encoding
        y_true_complement = Lambda(lambda arg: (K.ones_like(arg) - arg))(y_true)
        y_true_onehot = concatenate([y_true_complement, y_true], axis=-1)

        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_true_onehot[..., c_t], K.floatx())
            final_mask += w * y_p * y_t

        loss = K.categorical_crossentropy(y_true_onehot, y_pred) * final_mask
        loss_flat = K.flatten(loss)
        self.metric = K.sum(loss_flat)
        return loss

    def loss_dice(self, y_true, y_pred):
        return self.dice_coef(y_true, y_pred)

    def metric_dice(self, y_true, y_pred):
        self.dice_coef(y_true, y_pred)
        return self.diceMetric

    def metric_dice_dist(self, y_true, y_pred):

        from keras import backend as K
        from keras.layers import Lambda

        y_true1 = Lambda((lambda arg: K.expand_dims(arg[..., 0], axis=-1)))(y_true)
        self.dice_coef(y_true1, y_pred)
        return self.diceMetric

    def dice_coef(self, y_true, y_pred):

        from keras import backend as K

        y_true_f = K.flatten(y_true)
        
        #y_pred was one-hot encoded so take only the second channel
        y_pred_f = K.flatten(y_pred[..., 1])
        intersection = K.sum(y_true_f * y_pred_f)
        self.diceMetric = (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)
        diceLoss = 1 - self.diceMetric
        
        return diceLoss

    def distTransErrorMatrix(self, y_true_onehot, y_dist, y_pred_max_mat):

        from keras import backend as K
        from keras.layers import Lambda, multiply, add, subtract

        y_p1 = Lambda((lambda arg: K.cast(arg[..., 1], K.floatx())))(y_pred_max_mat)
        y_t0 = Lambda((lambda arg: K.cast(arg[..., 0], K.floatx())))(y_true_onehot)
        y_t1 = Lambda((lambda arg: K.cast(arg[..., 1], K.floatx())))(y_true_onehot)
        y_d = Lambda((lambda arg: K.cast(arg[..., 0], K.floatx())))(y_dist)

        #squash, invert distance map, multiply/exponentiate by some factor
        factor = 2.0
        currMin = 1.0 / (factor + 1.0)
        currMax = 1.0
        newMin = 1.0
        newMax = currMin
        y_d_squashed = Lambda(lambda arg: (arg / (arg + (factor * K.ones_like(arg)) + K.epsilon())))(y_d)
        y_d_invert = Lambda(lambda arg: ((((newMax - newMin) * (arg - currMin)) / (currMax - currMin)) + newMin))(y_d_squashed)
        y_d = Lambda(lambda arg: (arg * (factor + 1.0)))(y_d_invert)

        # Interior error
        multLayer = multiply([y_t1, y_p1])
        subtractLayer = subtract([y_t1, multLayer])
        interiorError = multiply([y_d, subtractLayer])

        # Exterior error
        multLayer2 = multiply([y_t0, y_p1])
        exteriorError = multiply([y_d, multLayer2])
        
        # Total error
        error = add([interiorError, exteriorError])
        
        #add one to error to ensure that aren't multiplying by zero
        error = Lambda(lambda arg: (K.ones_like(arg) + arg))(error)
        return error

    def metric_acc(self, y_true, y_pred):

        import keras
        from keras import backend as K
        from keras.layers import Lambda

        # Only need first channel of y_true!
        y_true1 = K.expand_dims(y_true[..., 0], axis=-1)
        y_true_complement = Lambda((lambda arg: K.ones_like(arg) - arg))(y_true1)
        y_true_onehot = keras.layers.concatenate([y_true_complement, y_true1], axis=-1)
        return keras.metrics.categorical_accuracy(y_true_onehot, y_pred)

    def init_f(self, shape, dtype=None):

        # Make 3D laplacian second derivative filter to find edges in binarized image
        ker = np.zeros(shape, dtype=dtype)
        ker[0, 1, 1] = 1
        ker[1, 1, 1] = -6
        ker[2, 1, 1] = 1
        ker[1, 0, 1] = 1
        ker[1, 2, 1] = 1
        ker[1, 1, 0] = 1
        ker[1, 1, 2] = 1

        return ker

    def smoothness(self, y_pred_max_mat):

        from keras import backend as K
        from keras.layers import Lambda, Conv3D

        # smoothness constraint
        y_p1 = Lambda((lambda arg: K.cast(arg, K.floatx())))(y_pred_max_mat)

        lap = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer=self.init_f, trainable=False)(y_p1)
        lap = K.abs(lap)
        lap_f = K.batch_flatten(lap[..., 0])

        # sum up the curvatures
        beta = K.sum(lap_f, axis=-1)
        return beta