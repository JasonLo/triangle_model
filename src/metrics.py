# %%

""" Custom metrics for diagnostic or experimental purpose"""

import tensorflow as tf
import pandas as pd


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
        """Item level calculation for evaluator"""
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

    def result(self):
        return self.out

    def reset_states(self):
        self.out.assign(0.0)

    def get_pho_idx_slot(self, act):
        """Trio function for getting phoneme 1 (Slot):
        Get cloest distance pho idx in a slot
        """
        slot_distance = tf.math.squared_difference(self.pho_map_values, act)
        sum_distance = tf.reduce_sum(slot_distance, -1)
        return tf.argmin(sum_distance)

    def get_pho_idx_item(self, act):
        """Trio function for getting phoneme 2 (Item):
        Get cloest distance pho idx in an item
        """
        act_2d = tf.reshape(tf.cast(act, tf.float32), shape=(10, 25))
        return tf.vectorized_map(self.get_pho_idx_slot, act_2d)

    @tf.function
    def get_pho_idx_batch(self, act):
        """Trio function for getting phoneme 3 (Batch):
        Get cloest distance pho idx in a batch
        """
        return tf.vectorized_map(self.get_pho_idx_item, act)


class RightSideAccuracy(tf.keras.metrics.Metric):
    """Accuracy based on all output nodes falls within the right half
    i.e. max(abs(true-pred)) < 0.5 is correct
    """

    def __init__(self, name="right_side_accuracy", **kwargs):
        super(RightSideAccuracy, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="right_side_accuracy", initializer="zeros")

    def update_state(self, y_true, y_pred):
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

    def reset_states(self):
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

    def result(self):
        return self.out

    def reset_states(self):
        self.out.assign(0.0)


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

    def reset_states(self):
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

    def reset_states(self):
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

    def reset_states(self):
        self.out.assign(0.0)


class OutputOfZeroTarget(tf.keras.metrics.Metric):
    """Export last slot average output in last batch of a epoch"""

    def __init__(self, name="output0", **kwargs):
        super(OutputOfZeroTarget, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="out0", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(tf.reduce_mean(y_pred[y_true == 0]))

    def result(self):
        return self.out

    def reset_states(self):
        self.out.assign(0.0)


class OutputOfOneTarget(tf.keras.metrics.Metric):
    """Export last slot average output in last batch of a epoch"""

    def __init__(self, name="output1", **kwargs):
        super(OutputOfOneTarget, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="out1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(tf.reduce_mean(y_pred[y_true == 1]))

    def result(self):
        return self.out

    def reset_states(self):
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

    def reset_states(self):
        self.out.assign(0.0)
