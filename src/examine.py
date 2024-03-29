import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import modeling, meta, data_wrangling


def expand_time_axis(x: tf.Tensor, n_timesteps: int) -> tf.Tensor:
    """Expand the time axis of a tensor."""
    x = tf.expand_dims(x, axis=0)
    return tf.tile(x, [n_timesteps, 1, 1])


def slice_last_dimension(x: tf.Tensor, idx: List[int]) -> tf.Tensor:
    """Slicing the last dimension of a tensor with a list of indices."""
    last_dim_start = idx[0]
    last_dim_size = len(idx)

    if tf.rank(x) == 3:  # Usually in full output [time, word, unit]
        return tf.slice(
            x, [0, 0, last_dim_start], [x.shape[0], x.shape[1], last_dim_size]
        )
    elif tf.rank(x) == 2:  # Usually in single tick output [word, unit]
        return tf.slice(x, [0, last_dim_start], [x.shape[0], last_dim_size])
    elif tf.rank(x) == 1:  # Usually in weights [unit]
        return tf.slice(x, [last_dim_start], [last_dim_size])
    else:
        raise ValueError(
            f"Unsupported rank {tf.rank(x)} tensor. Only works for rank 2 and 3."
        )


def get_weights(model: tf.keras.Model, name: str) -> tf.Tensor:
    """Get the weights of a layer from a TF model."""
    return [w for w in model.weights if w.name.endswith(f"{name}:0")][0]


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
        self.y_pred = self.model(expand_time_axis(x, self.cfg.n_timesteps))

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

    def weigths_and_biases(self, output_layer, units: List[int] = None) -> dict:
        """Get the mean weights and biases of a output layer."""
        if output_layer == "pho":

            op = get_weights(self.model, "w_hop_hp") * 0.5
            cp = get_weights(self.model, "w_cp") * 0.5
            sp = get_weights(self.model, "w_hsp_hp") * 0.5
            bias_p = get_weights(self.model, "bias_p")

            if units is not None:
                op = slice_last_dimension(op, units)
                cp = slice_last_dimension(cp, units)
                sp = slice_last_dimension(sp, units)
                bias_p = slice_last_dimension(bias_p, units)

            return {
                "op": op.numpy().mean(),
                "cp": cp.numpy().mean(),
                "sp": sp.numpy().mean(),
                "bias_p": bias_p.numpy().mean(),
            }

        elif output_layer == "sem":
            pass

    @property
    def mean_bias(self) -> None:
        """Get the mean bias of the model."""
        bias_p = get_weights(self.model, "bias_p")
        bias_s = get_weights(self.model, "bias_s")
        return {"PHO": bias_p.numpy().mean(), "SEM": bias_s.numpy().mean()}

    @property
    def mean_first_tick(self) -> None:
        """Get the first tick value from weights."""
        op = get_weights(self.model, "w_hop_hp") * 0.5
        cp = get_weights(self.model, "w_cp") * 0.5
        sp = get_weights(self.model, "w_hsp_hp") * 0.5
        os = get_weights(self.model, "w_hos_hs") * 0.5
        cs = get_weights(self.model, "w_cs") * 0.5
        ps = get_weights(self.model, "w_hps_hs") * 0.5
        return {
            "OP": op.numpy().mean(),
            "CP": cp.numpy().mean(),
            "SP": sp.numpy().mean(),
            "OS": os.numpy().mean(),
            "CS": cs.numpy().mean(),
            "PS": ps.numpy().mean(),
        }

    def get_input_ticks(self, name: str, act: int, units: List[int] = None) -> np.array:
        """Get the mean of a variable over a time tick.

        Arguments:
            name (str): The name of the variable.
            act (int): The target activation mask, 1/0.
            units (list): Index of subset units.
        """
        output_layer = self._guess_output_layer(name)
        pred = self.y_pred[name]
        n_ticks = pred.numpy().shape[0]
        y_true = expand_time_axis(self.testset[output_layer], n_ticks)

        # if act:  # Has target activation mask
        target = tf.cast(tf.equal(y_true, act), tf.float32)
        # else:
        # target = tf.ones_like(y_true)

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

    def plot_input(
        self,
        task,
        epoch,
        act=1,
        pho_units=None,
        sem_units=None,
        save=False,
        ylim=(-20, 20),
    ):
        """Plot the input temporal dynamics in SEM and PHO by target activation."""
        self.restore_epoch(epoch)
        self.model.set_active_task(task)
        self.eval(task)

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
        fig.suptitle(f"Task: {task}, Epoch: {epoch}, Target activation: {act}")

        # Plot phonology as output.
        axs[0].title.set_text(f"Input to PHO (target = {act})")
        axs[0].plot([0] * 13, color="black")

        input_op = self.get_input_ticks("input_hop_hp", act, pho_units)
        input_sp = self.get_input_ticks("input_hsp_hp", act, pho_units)
        input_cp = self.get_input_ticks("input_cpp_cp", act, pho_units)
        sum_input_pho = self.get_input_ticks("input_pho", act, pho_units)

        axs[0].plot(input_op, label="op", linestyle="--")
        axs[0].plot(input_sp, label="sp", linestyle="--")
        axs[0].plot(input_cp, label="cp", linestyle="--")
        axs[0].plot(sum_input_pho, label="pho")
        axs[0].legend()

        # Plot semantic as output.
        axs[1].title.set_text(f"Input to SEM (target = {act})")
        axs[1].plot([0] * 13, color="black")

        input_os = self.get_input_ticks("input_hos_hs", act, sem_units)
        input_ps = self.get_input_ticks("input_hps_hs", act, sem_units)
        input_cs = self.get_input_ticks("input_css_cs", act, sem_units)
        sum_input_sem = self.get_input_ticks("input_sem", act, sem_units)

        axs[1].plot(input_os, label="os", linestyle="--")
        axs[1].plot(input_ps, label="ps", linestyle="--")
        axs[1].plot(input_cs, label="cs", linestyle="--")
        axs[1].plot(sum_input_sem, label="sem")
        axs[1].legend()

        fig.set_facecolor("w")

        if ylim:
            axs[0].set_ylim(ylim)
            axs[1].set_ylim(ylim)

        if save:
            os.makedirs(f"{self.cfg.plot_folder}/temporal_dynamic", exist_ok=True)
            plt.savefig(
                f"{self.cfg.plot_folder}/temporal_dynamic/{task}_e{epoch}_a{act}.png"
            )
