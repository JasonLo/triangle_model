import tensorflow as tf
""" Custom metrics for diagnostic or experimental purpose"""


class NodeCounter(tf.keras.metrics.Metric):
    """Export last slot average output in a batch
    """

    def __init__(self, name="node_counter", **kwargs):
        super(NodeCounter, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(name="node_counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_sum(
                tf.cast(
                    tf.math.greater_equal(tf.math.abs(
                        y_pred - y_true), 0), tf.float32,
                )
            )
        )

    def result(self):
        return self.out

    def reset_states(self):
        self.out.assign(0.0)


class ZERCount(tf.keras.metrics.Metric):
    """Count the number of node that has < 0.1 Absolute error
    """

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
    """Count the number of node that has > 0.9 Absolute error
    """

    def __init__(self, name="zer_wrong_counter", **kwargs):
        super(ZERWrongSideCount, self).__init__(name=name, **kwargs)
        self.out = self.add_weight(
            name="zer_wrong_counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.out.assign(
            tf.reduce_sum(
                tf.cast(
                    tf.math.greater(
                        tf.math.abs(y_pred - y_true), 0.9
                    ),
                    tf.float32,
                )
            )
        )

    def result(self):
        return self.out

    def reset_states(self):
        self.out.assign(0.0)


class OutputOfZeroTarget(tf.keras.metrics.Metric):
    """Export last slot average output in last batch of a epoch
    """

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
    """Export last slot average output in last batch of a epoch
    """

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
        input_ones = -tf.math.log(1./act_ones - 1.)
        self.out.assign(tf.reduce_mean(input_ones))

    def result(self):
        return self.out

    def reset_states(self):
        self.out.assign(0.0)

