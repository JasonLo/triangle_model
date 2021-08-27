import os, argparse
from tqdm import tqdm
import tensorflow as tf
import meta, data_wrangling, metrics, modeling, benchmark_hs04
import environment as env


def main(json_file: str):
    """Run a TF modeling training with json config file
    Changing CPU visibility doesn't work on bash
    Workaround on bash:
    CUDA_VISIBLE_DEVICES=x python3 quick_run.py -f "path_to_json"
    """

    cfg = meta.Config.from_json(json_file)

    # Build all supporting elements
    tf.random.set_seed(cfg.rng_seed)
    data = data_wrangling.MyData()

    model = modeling.MyModel(cfg)
    model.build()

    experience = env.Experience.from_config(cfg.environment_config)
    sampler = env.Sampler(cfg, data, experience)
    batch_generator = sampler.generator()

    # Load pretraining
    weight_file = os.path.join("models", "Perfect_pretrain", "weights", "ep3000")
    model.load_weights(weight_file)

    optimizers = {}
    loss_fns = {}
    train_losses = {}  # Mean loss (only for TensorBoard)
    train_metrics = {}

    # Task specific accuracy
    ## Caution PhoAccuracy is stateful (only taking last batch value in an epoch)
    ## Otherwise, all Stateless metrics are the average of all batches within an epoch

    acc = {"pho": metrics.PhoAccuracy, "sem": metrics.StatelessRightSideAccuracy}
    sse = metrics.StatelessSumSquaredError

    for task in cfg.task_names:
        # optimizers[task] = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate)
        optimizers[task] = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        if cfg.zero_error_radius is not None:
            loss_fns[task] = metrics.CustomBCE(radius=cfg.zero_error_radius)
        else:
            loss_fns[task] = tf.keras.losses.BinaryCrossentropy()

        train_losses[task] = tf.keras.metrics.Mean(
            f"train_loss_{task}", dtype=tf.float32
        )  # for tensorboard only

        task_output = modeling.IN_OUT[task][1]

        if type(task_output) is list:
            train_metrics[task] = {}

            for out in task_output:
                train_metrics[task][out] = [
                    acc[out](f"{task}_{out}_acc"),
                    sse(f"{task}_{out}_sse"),
                ]
        else:
            train_metrics[task] = [acc[task_output](f"{task}_acc"), sse(f"{task}_sse")]

    def get_train_step(task):
        input_name, output_name = modeling.IN_OUT[task]

        if task == "triangle":

            @tf.function()
            def train_step(
                x, y, model, task, loss_fn, optimizer, train_metrics, train_losses
            ):
                """Train a batch, log loss and metrics (last time step only)"""

                train_weights_name = [
                    x + ":0" for x in modeling.WEIGHTS_AND_BIASES[task]
                ]
                train_weights = [
                    x for x in model.weights if x.name in train_weights_name
                ]

                # TF Automatic differentiation
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    # training flag can be access within model by K.in_train_phase()
                    # it can change the behavior in model() (e.g., turn on/off noise)

                    loss_value_pho = loss_fn(y["pho"], y_pred["pho"])
                    loss_value_sem = loss_fn(y["sem"], y_pred["sem"])
                    loss_value = loss_value_pho + loss_value_sem

                grads = tape.gradient(loss_value, train_weights)

                # Weight update
                optimizer.apply_gradients(zip(grads, train_weights))
                train_losses.update_state(loss_value)

        else:  # Single output tasks

            @tf.function()
            def train_step(
                x, y, model, task, loss_fn, optimizer, train_metrics, train_losses
            ):
                train_weights_name = [
                    x + ":0" for x in modeling.WEIGHTS_AND_BIASES[task]
                ]
                train_weights = [
                    x for x in model.weights if x.name in train_weights_name
                ]

                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss_value = loss_fn(y, y_pred[output_name])

                grads = tape.gradient(loss_value, train_weights)
                optimizer.apply_gradients(zip(grads, train_weights))
                train_losses.update_state(loss_value)

        return train_step

    train_steps = {task: get_train_step(task) for task in cfg.task_names}

    def write_weight_histogram_to_tensorboard(step):
        """Weight histogram"""
        [tf.summary.histogram(f"{x.name}", x, step=step) for x in model.weights]

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(cfg.tensorboard_folder, "train")
    )

    for epoch in tqdm(range(cfg.total_number_of_epoch)):
        for step in range(cfg.steps_per_epoch):
            # Draw task, create batch
            task, exposed_words_idx, x_batch_train, y_batch_train = next(
                batch_generator
            )

            # task switching must be done outside train_step function (will crash otherwise)
            model.set_active_task(task)

            # Run a train step
            train_steps[task](
                x_batch_train,
                y_batch_train,
                model,
                task,
                loss_fns[task],
                optimizers[task],
                train_metrics[task],
                train_losses[task],
            )

        with train_summary_writer.as_default():
            write_weight_histogram_to_tensorboard(step=epoch)
            [train_losses[x].reset_states() for x in cfg.task_names]

        ## Save weights
        one_indexing_epoch = epoch + 1
        if one_indexing_epoch in cfg.saved_epochs:
            weight_path = cfg.saved_weights_fstring.format(epoch=one_indexing_epoch)
            model.save_weights(weight_path, overwrite=True, save_format="tf")

    # Run benchmarks
    benchmark_hs04.main(cfg.code_name)


if __name__ == "__main__":
    """Command line entry point, take code_name and testcase to run tests"""
    parser = argparse.ArgumentParser(description="Train TF model with config json")
    parser.add_argument("-f", "--json_file", required=True, type=str)
    args = parser.parse_args()
    main(args.json_file)
