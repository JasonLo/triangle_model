import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper
import pandas as pd
import numpy as np

# Loss
class CustomBCE(tf.keras.losses.Loss):
    """Binarycross entropy loss with variable zero-error-radius"""

    def __init__(self, radius=0.1, name="bce_with_ZER"):
        super().__init__(name=name)
        self.radius = radius
        self.sample_weights = None

    def set_sample_weights(self, sample_weights):
        """Method to set sample_weights for scaling losses
        Cannot pass as an argument in call(), this is a work around
        call will automatically scale losses if sample_weight exists
        """
        self.sample_weights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)

    def disable_losses_scaling(self):
        self.sample_weights = None

    @staticmethod
    def zer_replace(target, output, zero_error_radius):
        """Replace output by target if value within zero-error-radius"""
        within_zer = tf.math.less_equal(
            tf.math.abs(output - target), tf.constant(zero_error_radius)
        )
        return tf.where(within_zer, target, output)

    @staticmethod
    def scale_losses(losses, sample_weights):
        """Multiply sample weights on losses"""
        timetick_dim = losses.shape[0]
        output_dim = losses.shape[2]

        expanded_sample_weights = tf.tile(
            sample_weights[tf.newaxis, :, tf.newaxis], [timetick_dim, 1, output_dim]
        )

        return losses * expanded_sample_weights

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Replace output by target if value within zero error radius
        zer_output = self.zer_replace(y_true, y_pred, self.radius)

        # Clip with a tiny constant to avoid zero division
        epsilon_ = tf.convert_to_tensor(K.epsilon(), y_pred.dtype)
        zer_output = tf.clip_by_value(zer_output, epsilon_, 1.0 - epsilon_)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.math.log(zer_output + K.epsilon())
        bce += (1 - y_true) * tf.math.log(1 - zer_output + K.epsilon())
        bce *= -1

        # Scale losses
        if self.sample_weights is not None:
            print(f"I have scaled sample weight by {self.sample_weights}")
            bce = self.scale_losses(bce, self.sample_weights)

        return bce


# Stateless metrics (mean of all batch within an epoch) for tensorboard
def sum_squared_error(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)


def right_side_accuracy(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, y_pred.dtype)
    return tf.reduce_min(tf.cast(tf.equal(y_true, y_pred), y_pred.dtype), axis=-1)


class StatelessRightSideAccuracy(MeanMetricWrapper):
    def __init__(self, name="stateless_right_side_acc", dtype=None):
        super().__init__(right_side_accuracy, name, dtype=dtype)


class StatelessSumSquaredError(MeanMetricWrapper):
    def __init__(self, name="stateless_sse", dtype=None):
        super().__init__(sum_squared_error, name, dtype=dtype)


# Stateful metrics
class PhoAccuracy(tf.keras.metrics.Metric):
    """Nearest phoneme based accuracy (HS04)"""

    def __init__(self, name="pho_accuracy", **kwargs):
        super(PhoAccuracy, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="pho_accuracy", initializer="zeros")

        # Load pho key
        pho_key_file = "/home/jupyter/tf/dataset/mappingv2.txt"
        mapping = pd.read_table(pho_key_file, header=None, delim_whitespace=True)
        pho_key = mapping.set_index(0).T.to_dict("list")

        self.pho_map_keys = tf.constant(list(pho_key.keys()))
        self.pho_map_values = tf.constant([v for v in pho_key.values()], tf.float32)

    def update_state(self, y_true, y_pred):
        """Batch level averaged metric"""
        self.out.assign(
            tf.reduce_mean(
                tf.cast(
                    tf.math.reduce_all(
                        tf.math.equal(
                            self.get_pho_idx_batch(y_pred),
                            self.get_pho_idx_batch(y_true),
                        ),
                        axis=-1,
                    ),
                    tf.float32,
                ),
                axis=-1,
            )
        )

    def item_metric(self, y_true, y_pred):
        """Item level calculation for evaluator
        y_true dims: (n items, pho dims)
        y_pred dims: (n items, pho dims)
        output: accuracy in each items, dim: (n items)
        """
        return tf.cast(
            tf.math.reduce_all(
                tf.math.equal(
                    self.get_pho_idx_batch(y_pred),
                    self.get_pho_idx_batch(y_true),
                ),
                axis=-1,
            ),
            tf.float32,
        ).numpy()

    def item_metric_multi_ans(self, y_trues, y_pred):
        """Check a predition match with any y_true pattern
        y_trues dims: (n items, n ans, pho dims)
        output dims: items
        """
        y_trues_idx = tf.vectorized_map(self.get_pho_idx_batch, y_trues)
        y_true_idx_t = tf.transpose(y_trues_idx, [1, 0, 2])
        y_pred_idx = self.get_pho_idx_batch(y_pred)
        eq = tf.vectorized_map(lambda x: tf.equal(y_pred_idx, x), y_true_idx_t)
        return tf.cast(tf.reduce_all(tf.reduce_any(eq, axis=0), axis=-1), tf.float32).numpy()

    def item_metric_multi_ans_ragged(self, y_trues_ragged, y_pred):
        """ Calculate acc with variable length answer keys in ragged tensor format"""
        y_pred_idx = self.get_pho_idx_batch(y_pred)
        output = np.empty(shape=(y_pred.shape[0],))
        # output = [self._ragged_eval_item(y_trues_ragged[i], y_pred_idx[i]) for i in range(y_pred_idx.shape[0])]

        for i, this_y_pred in enumerate(y_pred_idx):
            this_y_trues_idx = tf.map_fn(self.get_pho_idx_item, y_trues_ragged[i], fn_output_signature=tf.int64)
            eq = tf.vectorized_map(lambda x: tf.equal(this_y_pred, x), this_y_trues_idx)
            eq_item = tf.reduce_all(eq, axis=-1)
            acc = tf.reduce_any(eq_item, axis=0)
            output[i] = acc.numpy()

            # print(this_y_pred)
            # print(this_y_trues_idx)
            # print(eq)
            # print(eq_item)
            # print(acc)
        return output           

    # def _ragged_eval_item(self, y_trues, y_pred):
    #     y_trues_idx = tf.map_fn(self.get_pho_idx_item, y_trues, fn_output_signature=tf.int64)
    #     eq = tf.map_fn(lambda x: tf.equal(y_pred, x), y_trues_idx, fn_output_signature=tf.bool)
    #     eq_item = tf.reduce_all(eq, axis=-1)
    #     acc = tf.reduce_any(eq_item, axis=0)
    #     return acc.numpy()

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)

    def get_pho_idx_slot(self, act):
        """Trio function for getting phoneme 1 (Slot):
        Get cloest distance pho idx in a slot
        Input shape expectation: (25 pho dims) i.e., one slot
        Output dim: (1)
        """
        slot_distance = tf.math.squared_difference(self.pho_map_values, act)
        sum_distance = tf.reduce_sum(slot_distance, -1)
        return tf.argmin(sum_distance)

    def get_pho_idx_item(self, act):
        """Trio function for getting phoneme 2 (Item):
        Get cloest distance pho idx in an item
        Input shape expectation: (250 pho dims) i.e., one item
        Output dim: (10 slots)
        """
        act_2d = tf.reshape(tf.cast(act, tf.float32), shape=(10, 25))
        return tf.vectorized_map(self.get_pho_idx_slot, act_2d)

    def get_pho_idx_batch(self, act):
        """Trio function for getting phoneme 3 (Batch):
        Get cloest distance pho idx in a batch
        Input shape expectation: (items, 250 pho dims)
        Output: (items, 10 slots)
        """
        return tf.vectorized_map(self.get_pho_idx_item, act)



class RightSideAccuracy(tf.keras.metrics.Metric):
    """Accuracy based on all output nodes falls within the right half
    i.e. max(abs(true-pred)) < 0.5 is correct
    """

    def __init__(self, name="right_side_accuracy", **kwargs):
        super(RightSideAccuracy, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="right_side_accuracy", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_mean(
                tf.cast(
                    tf.math.less(
                        tf.reduce_max(tf.math.abs(y_pred - y_true), axis=-1), 0.5
                    ),
                    tf.float32,
                ),
                axis=-1,
            )
        )

    def item_metric(self, y_true, y_pred):
        return tf.cast(
            tf.math.less(tf.reduce_max(tf.math.abs(y_pred - y_true), axis=-1), 0.5),
            tf.float32,
        ).numpy()

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


class SumSquaredError(tf.keras.metrics.Metric):
    """Accuracy based on all output nodes falls within the right half
    i.e. max(abs(true-pred)) < 0.5 is correct
    """

    def __init__(self, name="sum_squared_error", **kwargs):
        super(SumSquaredError, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="sum_squared_error", initializer="zeros")

    def update_state(self, y_true, y_pred):
        self.out.assign(
            tf.reduce_mean(
                tf.cast(
                    tf.reduce_sum(tf.math.square(y_pred - y_true), axis=-1),
                    tf.float32,
                ),
                axis=-1,
            )
        )

    def item_metric(self, y_true, y_pred):
        return tf.cast(
            tf.reduce_sum(tf.math.square(y_pred - y_true), axis=-1),
            tf.float32,
        ).numpy()


    def item_metric_multi_ans(self, y_trues, y_pred):
        """Check a predition match with any y_true pattern
        y_trues dims: (n items, n ans, pho dims)
        output dims: items
        """
        y_trues_idx = tf.vectorized_map(self.get_pho_idx_batch, y_trues)
        y_trues_idx_t = tf.transpose(y_trues_idx, [1, 0, 2])
        y_pred_idx = self.get_pho_idx_batch(y_pred)
        eq = tf.vectorized_map(lambda x: tf.equal(y_pred_idx, x), y_trues_idx_t)
        return tf.cast(tf.reduce_all(tf.reduce_any(eq, axis=0), axis=-1), tf.float32).numpy()


    def item_metric_multi_ans(self, y_trues, y_pred):
        # rearrange dims from (item, ans, node) to (ans, item, node)
        y_trues_t = tf.transpose(y_trues, [1, 0, 2])  
        sse = tf.vectorized_map(lambda x: tf.reduce_sum(tf.square(y_pred - x), axis=-1), y_trues_t)
        
        # Min SSE to all possible answers in each item
        return tf.cast(tf.reduce_min(sse, axis=0), tf.float32).numpy()



    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


####  Experimental metrics ####
class NodeCounter(tf.keras.metrics.Metric):
    """Export last slot average output in a batch"""

    def __init__(self, name="node_counter", **kwargs):
        super(NodeCounter, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="node_counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_sum(
                tf.cast(
                    tf.math.greater_equal(tf.math.abs(y_pred - y_true), 0),
                    tf.float32,
                )
            )
        )

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


class ZERCount(tf.keras.metrics.Metric):
    """Count the number of node that has < 0.1 Absolute error"""

    def __init__(self, name="zer_counter", **kwargs):
        super(ZERCount, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="zer_counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_sum(
                tf.cast(
                    tf.math.less(tf.math.abs(y_pred - y_true), 0.1),
                    tf.float32,
                )
            )
        )

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


class ZERWrongSideCount(tf.keras.metrics.Metric):
    """Count the number of node that has > 0.9 Absolute error"""

    def __init__(self, name="zer_wrong_counter", **kwargs):
        super(ZERWrongSideCount, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="zer_wrong_counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_sum(
                tf.cast(
                    tf.math.greater(tf.math.abs(y_pred - y_true), 0.9),
                    tf.float32,
                )
            )
        )

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


class OutputOfZeroTarget(tf.keras.metrics.Metric):
    """Export last slot average output in last batch of a epoch"""

    def __init__(self, name="output0", **kwargs):
        super(OutputOfZeroTarget, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="out0", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_sum(y_pred * (1 - y_true)) / tf.reduce_sum(1 - y_true)
        )

    def item_metric(self, y_true, y_pred):
        numer = tf.reduce_sum((y_pred * (1 - y_true)), axis=-1)
        denom = tf.cast(tf.reduce_sum((1 - y_true), axis=-1), tf.float32)
        act0 = numer / denom
        return act0.numpy()

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


class OutputOfOneTarget(tf.keras.metrics.Metric):
    """Export last slot average output in last batch of a epoch"""

    def __init__(self, name="output1", **kwargs):
        super(OutputOfOneTarget, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="out1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(tf.reduce_sum(y_pred * y_true) / tf.reduce_sum(y_true))

    def item_metric(self, y_true, y_pred):
        numer = tf.reduce_sum((y_pred * y_true), axis=-1)
        denom = tf.cast(tf.reduce_sum(y_true, axis=-1), tf.float32)
        act1 = numer / denom
        return act1.numpy()

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)


class InputOfOneTarget(tf.keras.metrics.Metric):
    def __init__(self, name="mean_input_of_one_target", **kwargs):
        super(InputOfOneTarget, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="mean_input_of_one_target", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        act_ones = y_pred[y_true == 1]
        input_ones = -tf.math.log(1.0 / act_ones - 1.0)
        self.out.assign(tf.reduce_mean(input_ones))

    def result(self):
        return self.out

    def reset_state(self):
        self.out.assign(0.0)
