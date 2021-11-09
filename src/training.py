import tensorflow as tf
from modeling import WEIGHTS_AND_BIASES, IN_OUT

def basic_train_step(task: str):
    """Construct a train step function for basic single output task."""
    input_name, output_name = IN_OUT[task]

    @tf.function()
    def train_step(x, y, model, task, loss_fn, optimizer, metrics, losses):
        train_weights_name = [x + ":0" for x in WEIGHTS_AND_BIASES[task]]
        train_weights = [x for x in model.weights if x.name in train_weights_name]

        # Automatic differentiation
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = loss_fn(y, y_pred[output_name])

        grads = tape.gradient(loss_value, train_weights)

        # Weights update
        optimizer.apply_gradients(zip(grads, train_weights))

        # TensorBoard mean loss
        losses.update_state(loss_value)

        # TensorBoard metrics (Counting last tick only)
        y_true = tf.cast(y[-1], tf.float32)
        y_pred = y_pred[output_name][-1]
        [m.update_state(y_true, y_pred) for m in metrics]

    return train_step


def triangle_train_step():
    """Construct a train step function for triangle task."""
    input_name, output_name = IN_OUT[task]
    
    @tf.function()
    def train_step(x, y, model, task, loss_fn, optimizer, metrics, losses):
        """Defines a step of training for triangle model."""

        train_weights_name = [x + ":0" for x in WEIGHTS_AND_BIASES["triangle"]]
        train_weights = [x for x in model.weights if x.name in train_weights_name]

        # Automatic differentiation
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value_pho = loss_fn(y["pho"], y_pred["pho"])
            loss_value_sem = loss_fn(y["sem"], y_pred["sem"])
            loss_value = loss_value_pho + loss_value_sem

        grads = tape.gradient(loss_value, train_weights)

        # Weights update
        optimizer.apply_gradients(zip(grads, train_weights))

        # TensorBoard mean loss
        losses.update_state(loss_value)

        # TensorBoard metrics (Counting last tick only)
        for output_name, metrics in metrics.items():
            y_true = tf.cast(y[output_name][-1], tf.float32)
            y_pred = y_pred[output_name][-1]
            [m.update_state(y_true, y_pred) for m in metrics]

    return train_step
