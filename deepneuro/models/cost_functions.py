import tensorflow as tf

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


def wasserstein_loss(discriminator, discriminator_loss, real_data, fake_data, batch_size, gradient_penalty_weight=10, name=''):

    # Implementation fo Wasserstein loss with gradient penalty.
    # I think some of this is more concise in updated tensorflow

    # Gradient Penalty from Wasserstein GAN GP, I believe? Check on it --andrew
    # Also investigate more of what's happening here --andrew
    differences = fake_data - real_data
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = real_data + (alpha * differences)
    _, discri_logits = discriminator(interpolates, name=name, reuse=True)
    gradients = tf.gradients(discri_logits, [interpolates])[0]

    # Some sort of norm from papers, check up on it. --andrew
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3, 4]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    # Update Loss functions..
    discriminator_loss += gradient_penalty_weight * gradient_penalty
    # discriminator_loss += 0.001 * tf.reduce_mean(tf.square(self.discriminator_2_real_logits - 0.0))

    return discriminator_loss