import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations, initializers, regularizers, Model
from tensorflow.keras.layers import Layer, Input
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output


# Zero-error-radius related:

from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import nn, clip_ops, math_ops, array_ops
from tensorflow.keras.backend import epsilon


def _constant_to_tensor(x, dtype):
    return constant_op.constant(x, dtype=dtype)


def _backtrack_identity(tensor):
    while tensor.op.type == "Identity":
        tensor = tensor.op.inputs[0]
    return tensor


def zer_replace(target, output, zero_error_radius):
    """Replace output by target if value within zero-error-radius
    """
    within_zer = tf.math.less_equal(tf.math.abs(
        output - target), tf.constant(zero_error_radius))
    return tf.where(within_zer, target, output)


class CustomBCE(keras.losses.Loss):
    """ Binarycross entropy loss with variable zero-error-radius
    """

    def __init__(self, radius=0.1, name="bce_with_ZER"):
        super().__init__(name=name)
        self.radius = radius

    def call(self, y_true, y_pred):
        if not isinstance(y_pred, (ops.EagerTensor, variables_module.Variable)):
            y_pred = _backtrack_identity(y_pred)

        # Replace output by target if value within zero error radius
        zer_output = zer_replace(y_true, y_pred, self.radius)

        # Clip with a tiny constant to avoid zero division
        epsilon_ = _constant_to_tensor(epsilon(), y_pred.dtype.base_dtype)
        zer_output = clip_ops.clip_by_value(
            zer_output, epsilon_, 1.0 - epsilon_)

        # Compute cross entropy from probabilities.
        bce = y_true * math_ops.log(zer_output + epsilon())
        bce += (1 - y_true) * math_ops.log(1 - zer_output + epsilon())
        return -bce




class ModelCheckpoint_custom(Callback):
    """
    Modified from original ModelCheckpoint
    Always save first 10 epochs regardless save period
    """

    def __init__(self, filepath, save_weights_only=False, period=1):
        super(ModelCheckpoint_custom, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (((epoch + 1) % self.period == 0) or epoch < 10):
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            clear_output(wait=True)
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)

    def on_train_end(self, logs=None):
        
