def cost_function_dict():

    return {'dice_coef': dice_coef,
            'dice_coef_loss': dice_coef_loss,
            'multi_dice_coef': multi_dice_coef,
            'multi_dice_loss': multi_dice_loss
            }


def dice_coef(y_true, y_pred, smooth=1.):

    from keras import backend as K

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
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