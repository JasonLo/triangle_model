import os
import tensorflow as tf
from tqdm import tqdm
from data_wrangling import MyData
from environment import Sampler
from modeling import WEIGHTS_AND_BIASES, IN_OUT, TriangleModel
from metrics import (
    CustomBCE,
    PhoAccuracy,
    StatelessRightSideAccuracy,
    StatelessSumSquaredError,
)


def basic_train_step(task: str):
    """Construct a train step function for basic single output task."""

    @tf.function()
    def train_step(x, y, model, task, loss_fn, optimizer, metrics, losses):
        """Defines a step of training for triangle model.
        Arguments:
            x: Input data.
            y: Target/Label data.
            model: Model to train.
            task: Task to train.
            loss_fn: Loss function.
            optimizer: Optimizer to use.
            metrics: Metrics to use.
            losses: TensorBoard mean losses.
        """
        input_name, output_name = IN_OUT[task]
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

    @tf.function()
    def train_step(x, y, model, task, loss_fn, optimizer, metrics, losses):
        """Defines a step of training for triangle model.
        Arguments:
            x: Input data.
            y: Target/Label data.
            model: Model to train.
            task: Task to train.
            loss_fn: Loss function.
            optimizer: Optimizer to use.
            metrics: Metrics to use.
            losses: TensorBoard mean losses.
        """
        input_name, output_name = IN_OUT["triangle"]
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

        # Update metrics (Counting last tick only)

        # PHO
        y_true_pho = tf.cast(y["pho"][-1], tf.float32)
        y_pred_pho = y_pred["pho"][-1]
        [m.update_state(y_true_pho, y_pred_pho) for m in metrics["pho"]]

        # SEM
        y_true_sem = tf.cast(y["sem"][-1], tf.float32)
        y_pred_sem = y_pred["sem"][-1]
        [m.update_state(y_true_sem, y_pred_sem) for m in metrics["sem"]]

    return train_step


class Trainer:
    """Trainer class for training a model.
    Since each sub-task has its own states, it must be trained with separate optimizer.
    """

    def __init__(self, cfg, experience):
        self.cfg = cfg
        self.experience = experience
        self.data = MyData()
        self.model = TriangleModel(cfg)
        self.model.build()
        self.sampler = Sampler(self.cfg, self.data, self.experience)
        self.batch_generator = self.sampler.generator()

        # Compiling
        self.optimizers = {}
        self.loss_fns = {}
        self.train_losses = {}
        self.train_metrics = {}
        self.train_steps = {}
        self.compile()

        # Checkpointing
        self.epoch = None
        self.checkpoint = None
        self.ckpt_manager = None
        self._create_checkpoint()
        if self.cfg.pretrain_checkpoint:
            self.load_model(self.cfg.pretrain_checkpoint)
            print(f"Loaded pretrain model from {self.cfg.pretrain_checkpoint}")

    @property
    def non_triangle_tasks_names(self):
        """Return the list of non-triangle tasks."""
        return [x for x in self.cfg.task_names if x != "triangle"]

    def compile(self):
        """Create the objects that needed for training."""

        for task in self.cfg.task_names:
            self._create_optimizer(task)
            self._create_loss(task)
            self._create_metrics(task)
            self._create_train_step(task)

    def _create_optimizer(self, task: str) -> None:
        """Create optimizer for a task."""
        if self.cfg.optimizer == "adam":
            self.optimizers[task] = tf.keras.optimizers.Adam(
                learning_rate=self.cfg.learning_rate
            )
        elif self.cfg.optimizer == "sgd":
            self.optimizers[task] = tf.keras.optimizers.SGD(
                learning_rate=self.cfg.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def _create_loss(self, task: str) -> None:
        """Create loss function and train losses for a task."""

        if self.cfg.zero_error_radius is not None:
            self.loss_fns[task] = CustomBCE(
                zero_error_radius=self.cfg.zero_error_radius
            )
        else:
            # Must use CustomBCE, keras BCE does not support loss ramping
            self.loss_fns[task] = CustomBCE(zero_error_radius=0.0)

        # Set losses time weights for loss ramping
        if self.cfg.loss_ramping:
            time_weights_all = [
                x / self.cfg.n_timesteps for x in range(self.cfg.n_timesteps + 1)
            ]
            time_weights = time_weights_all[-self.cfg.inject_error_ticks :]
            self.loss_fns[task].set_time_weights(time_weights)

        # For TensorBoard
        self.train_losses[task] = tf.keras.metrics.Mean(
            f"train_loss_{task}", dtype=tf.float32
        )

    def _create_metrics(self, task: str) -> None:
        """Create metrics for a task.
        Due to performance reasons, PHO accuracy is using a stateful metric.
            - Stateless metrics: the average metric of all batches within an epoch (semantic acc, sse)
            - Stateful metrics: only taking last step/batch value in an epoch (pho acc)
        """
        acc = {"pho": PhoAccuracy, "sem": StatelessRightSideAccuracy}
        sse = StatelessSumSquaredError

        task_output = IN_OUT[task][1]
        if task == "triangle":
            self.train_metrics[task] = {}

            # PHO output
            self.train_metrics[task]["pho"] = [
                acc["pho"](name=f"{task}_pho_acc"),
                sse(name=f"{task}_pho_sse"),
            ]

            # SEM output
            self.train_metrics[task]["sem"] = [
                acc["sem"](name=f"{task}_sem_acc"),
                sse(name=f"{task}_sem_sse"),
            ]

        else:
            self.train_metrics[task] = [
                acc[task_output](name=f"{task}_acc"),
                sse(name=f"{task}_sse"),
            ]

    def _create_train_step(self, task: str) -> None:
        """Create train step for a task."""
        if task == "triangle":
            self.train_steps[task] = triangle_train_step()
        else:
            self.train_steps[task] = basic_train_step(task)

    def _create_checkpoint(self) -> None:
        """Create checkpoint."""

        self.epoch = tf.Variable(0, name="epoch", dtype=tf.int64)

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizers, model=self.model, epoch=self.epoch
        )

        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.cfg.checkpoint_folder,
            max_to_keep=None,  # Keep all checkpoints
            checkpoint_name="epoch",
        )

    def load_model(self, checkpoint_path: str) -> None:
        """Restore model from a specific checkpoint.
        Use it to load pretrained model.
        """
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(checkpoint_path).expect_partial()
        print(f"Model restored from {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restore model, epoch, optimizers from a specific checkpoint.
        Use it to resume from unfinished training.
        """
        self.checkpoint.restore(checkpoint_path)
        print(f"Model, Epoch, and Optimizers restored from {checkpoint_path}")

    def try_to_resume(self) -> None:
        """Try to resume training from latest checkpoint.
        Restore includes model, optimizers, and epoch.
        """
        latest_checkpoint = self.ckpt_manager.latest_checkpoint

        if latest_checkpoint:
            self.load_checkpoint(latest_checkpoint)
        else:
            print("Initializing from scratch.")

    def _collect_accuracy_status(self) -> str:
        """Collect accuracy metrics status for reporting in progress bar."""
        status = []

        if "triangle" in self.cfg.task_names:
            tri_pho = self.train_metrics["triangle"]["pho"][0].result()
            tri_sem = self.train_metrics["triangle"]["sem"][0].result()
            status.append(f"Triangle PHO ACC: {tri_pho:.2f}")
            status.append(f"Triangle SEM ACC: {tri_sem:.2f}")

        for task in self.non_triangle_tasks_names:
            acc = self.train_metrics[task][0].result()
            status.append(f"{task} ACC: {acc:.2f}")

        return " | ".join(status)

    def _reset_all_losses_and_metrics(self) -> None:
        """Reset all losses and metrics."""

        # Reset losses in all tasks
        [self.train_losses[x].reset_states() for x in self.cfg.task_names]

        # Reset metrics in triangle task
        if "triangle" in self.cfg.task_names:
            self.train_metrics["triangle"]["pho"][0].reset_states()  # PHO acc
            self.train_metrics["triangle"]["pho"][1].reset_states()  # PHO sse
            self.train_metrics["triangle"]["sem"][0].reset_states()  # SEM acc
            self.train_metrics["triangle"]["sem"][1].reset_states()  # SEM sse

        # Reset metrics in all other tasks
        for task in self.non_triangle_tasks_names:
            self.train_losses[task].reset_states()

    def train(self, tensorboard_manager=None, try_to_resume=True) -> None:

        if try_to_resume:
            self.try_to_resume()

        progress_bar = tqdm(total=self.cfg.total_number_of_epoch, desc="Training")
        progress_bar.update(self.epoch.numpy())

        while self.epoch.numpy() < self.cfg.total_number_of_epoch:

            # Train an epoch
            for step in range(self.cfg.steps_per_epoch):
                # Draw task, create batch
                (
                    task,
                    exposed_words_idx,
                    exposed_word,
                    x_batch_train,
                    y_batch_train,
                ) = next(self.batch_generator)

                # task switching must be done outside train_step function (will crash otherwise)
                self.model.set_active_task(task)
                # Run a train step
                self.train_steps[task](
                    x=x_batch_train,
                    y=y_batch_train,
                    model=self.model,
                    task=task,
                    loss_fn=self.loss_fns[task],
                    optimizer=self.optimizers[task],
                    metrics=self.train_metrics[task],
                    losses=self.train_losses[task],
                )

            # Post epoch Ops
            # Write Tensorboard
            if tensorboard_manager:
                tensorboard_manager.write(step=self.epoch.numpy())

            # Update progress bar
            progress_bar.set_postfix_str(self._collect_accuracy_status())
            progress_bar.update(1)
            self.epoch.assign_add(1)

            # Reset metric and loss
            self._reset_all_losses_and_metrics()

            if self.epoch.numpy() in self.cfg.saved_epochs:
                self.ckpt_manager.save(self.epoch)


################################## TENSORBOARD ##################################


class TensorBoardManager:
    def __init__(self, cfg, model, metrics, losses):
        self.cfg = cfg
        self.model = model
        self.metrics = metrics
        self.losses = losses
        self.writer = tf.summary.create_file_writer(
            logdir=os.path.join(cfg.tensorboard_folder, "train")
        )

    def write(self, step: int):
        """Write everything to TensorBoard."""
        with self.writer.as_default():
            self.write_weights_to_tensorboard(step)
            for task in self.cfg.task_names:
                self.write_loss_to_tensorboard(task, step)
                self.write_metrics_to_tensorboard(task, step)

    def write_loss_to_tensorboard(self, task: str, step: int):
        """Write metrics and loss to tensorboard"""
        loss = self.losses[task]
        tf.summary.scalar(loss.name, loss.result(), step=step)

    def write_metrics_to_tensorboard(self, task: str, step: int):
        """Write metrics to tensorboard"""

        if task == "triangle":
            pho_acc = self.metrics["triangle"]["pho"][0].result()
            pho_sse = self.metrics["triangle"]["pho"][1].result()
            sem_acc = self.metrics["triangle"]["sem"][0].result()
            sem_sse = self.metrics["triangle"]["sem"][1].result()

            tf.summary.scalar("tri_pho_acc", pho_acc, step=step)
            tf.summary.scalar("tri_pho_sse", pho_sse, step=step)
            tf.summary.scalar("tri_sem_acc", sem_acc, step=step)
            tf.summary.scalar("tri_sem_sse", sem_sse, step=step)

        else:
            acc = self.metrics[task][0].result()
            sse = self.metrics[task][1].result()
            tf.summary.scalar(f"{task}_acc", acc, step=step)
            tf.summary.scalar(f"{task}_sse", sse, step=step)

    def write_weights_to_tensorboard(self, step: int):
        """Weights for histogram."""
        [tf.summary.histogram(f"{x.name}", x, step=step) for x in self.model.weights]
