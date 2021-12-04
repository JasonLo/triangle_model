import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import modeling, meta, data_wrangling


def slice_last_dimension(x: tf.Tensor, idx: List[int]) -> tf.Tensor:
    """Slicing the last dimension of a tensor with a list of indices."""
    last_dim_start = idx[0]
    last_dim_size = len(idx)
    return tf.slice(x, [0, 0, last_dim_start], [x.shape[0], x.shape[1], last_dim_size])


def get_weights(model: tf.keras.Model, name: str) -> tf.Tensor:
    """Get the weights of a layer from a TF model."""
    return [w.numpy() for w in model.weights if w.name.endswith(f"{name}:0")][0]


class Examine:
    """Examine the input temporal dynamics of a model."""

    def __init__(self, cfg: meta.Config) -> None:
        self.cfg = cfg
        self.model = modeling.TriangleModel(cfg)
        self.model.build()
        self.testset = data_wrangling.load_testset("train_r100")
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def restore_epoch(self, epoch: int) -> None:
        """Restore the model from a checkpoint."""
        self.checkpoint.restore(
            f"{self.cfg.checkpoint_folder}/epoch-{epoch}"
        ).expect_partial()

    def eval(self, task) -> None:
        """Compute y_pred from testset."""
        input_name = modeling.IN_OUT[task][0]
        x = self.testset[input_name]
        self.y_pred = self.model([x] * self.cfg.n_timesteps)

    @staticmethod
    def _guess_output_layer(name: str) -> str:
        """Guess the output layer from the input name."""
        if (name[-1] == "p") or (name[-3:] == "pho"):
            output_layer = "pho"
        elif (name[-1] == "s") or (name[-3:] == "sem"):
            output_layer = "sem"
        else:
            raise ValueError(f"{name} is not connecting to SEM/PHO layer.")
        return output_layer

    def get_input_ticks(
        self, name: str, target_mask: int, units: List[int] = None
    ) -> np.array:
        """Get the mean of a variable over a time tick.
        Args:
            name (str): The name of the variable.
            target_mask (int): The target activation mask, 1/0.
            units (list): Index of subset units.
        """
        output_layer = self._guess_output_layer(name)
        pred = self.y_pred[name]
        n_ticks = pred.numpy().shape[0]

        y_true = tf.expand_dims(self.testset[output_layer], axis=0)
        y_true = tf.tile(y_true, [n_ticks, 1, 1])
        target = tf.cast(tf.equal(y_true, target_mask), tf.float32)

        if units is not None:  # Subset unit axis.
            pred = slice_last_dimension(pred, units)
            y_true = slice_last_dimension(y_true, units)
            target = slice_last_dimension(target, units)

        n_target_per_tick = tf.reduce_sum(target, axis=(1, 2))
        masked_pred = pred * tf.cast(target, tf.float32)
        mean_input_per_tick = (
            tf.reduce_sum(masked_pred, axis=(1, 2)) / n_target_per_tick
        )
        return mean_input_per_tick.numpy()

    def plot_input(self, task, epoch, pho_units=None, sem_units=None):
        """Plot the input temporal dynamics in SEM and PHO by target activation."""
        self.restore_epoch(epoch)
        self.model.set_active_task(task)
        self.eval(task)

        fig, axs = plt.subplots(2, 2, sharey=True, figsize=(15, 10))
        axs[0, 0].title.set_text("Input to SEM (target = 1)")
        axs[0, 0].plot([0] * 13, color="black")
        axs[0, 0].plot(
            self.get_input_ticks("input_hos_hs", 1, sem_units),
            label="os",
            linestyle="--",
        )
        axs[0, 0].plot(
            self.get_input_ticks("input_hps_hs", 1, sem_units),
            label="ps",
            linestyle="--",
        )
        axs[0, 0].plot(
            self.get_input_ticks("input_css_cs", 1, sem_units),
            label="cs",
            linestyle="--",
        )
        axs[0, 0].plot(self.get_input_ticks("input_sem", 1, sem_units), label="sem")
        axs[0, 0].legend()

        axs[1, 0].title.set_text("Input to PHO (target = 1)")
        axs[1, 0].plot([0] * 13, color="black")
        axs[1, 0].plot(
            self.get_input_ticks("input_hop_hp", 1, pho_units),
            label="op",
            linestyle="--",
        )
        axs[1, 0].plot(
            self.get_input_ticks("input_hsp_hp", 1, pho_units),
            label="sp",
            linestyle="--",
        )
        axs[1, 0].plot(
            self.get_input_ticks("input_cpp_cp", 1, pho_units),
            label="cp",
            linestyle="--",
        )
        axs[1, 0].plot(self.get_input_ticks("input_pho", 1, pho_units), label="pho")
        axs[1, 0].legend()

        axs[0, 1].title.set_text("Input to SEM (target = 0)")
        axs[0, 1].plot([0] * 13, color="black")
        axs[0, 1].plot(
            self.get_input_ticks("input_hos_hs", 0, sem_units),
            label="os",
            linestyle="--",
        )
        axs[0, 1].plot(
            self.get_input_ticks("input_hps_hs", 0, sem_units),
            label="ps",
            linestyle="--",
        )
        axs[0, 1].plot(
            self.get_input_ticks("input_css_cs", 0, sem_units),
            label="cs",
            linestyle="--",
        )
        axs[0, 1].plot(self.get_input_ticks("input_sem", 0, sem_units), label="sem")
        axs[0, 1].legend()

        axs[1, 1].title.set_text("Input to PHO (target = 0)")
        axs[1, 1].plot([0] * 13, color="black")
        axs[1, 1].plot(
            self.get_input_ticks("input_hop_hp", 0, pho_units),
            label="op",
            linestyle="--",
        )
        axs[1, 1].plot(
            self.get_input_ticks("input_hsp_hp", 0, pho_units),
            label="sp",
            linestyle="--",
        )
        axs[1, 1].plot(
            self.get_input_ticks("input_cpp_cp", 0, pho_units),
            label="cp",
            linestyle="--",
        )
        axs[1, 1].plot(self.get_input_ticks("input_pho", 0, pho_units), label="pho")
        axs[1, 1].legend()
        fig.set_facecolor("w")
