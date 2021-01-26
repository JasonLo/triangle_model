# %%
import sys
import time

import numpy as np
import pandas as pd
from pandas.io.stata import stata_epoch
import tensorflow as tf
from IPython.display import clear_output
from tqdm import tqdm

sys.path.append("/home/jupyter/tf/src")
import data_wrangling
import meta
import metrics
import modeling

# %% Parameters block

code_name = "boo"
tf_root = "/home/jupyter/tf"

# Model architechture
ort_units = 119
pho_units = 250
sem_units = 2446

hidden_os_units = 500
hidden_op_units = 100
hidden_ps_units = 500
hidden_sp_units = 500

pho_cleanup_units = 50
sem_cleanup_units = 50

pho_noise_level = 0.0
sem_noise_level = 0.0

activation = "sigmoid"
tau = 1 / 3
max_unit_time = 4.0
output_ticks = 4

# Training
sample_name = "hs04"
rng_seed = 53797
learning_rate = 0.01
n_mil_sample = 0.1
batch_size = 100
save_freq = 10


# %% Package model configurations into meta.ModelConfig()
config_dict = {}

for v in meta.CORE_CONFIGS:
    try:
        config_dict[v] = globals()[v]
    except:
        raise

for v in meta.OPTIONAL_CONFIGS:
    try:
        config_dict[v] = globals()[v]
    except:
        pass

# Construct ModelConfig object
cfg = meta.ModelConfig(**config_dict)
cfg.save()
del config_dict

# %% Build model and all supporting components
tf.random.set_seed(cfg.rng_seed)
data = data_wrangling.MyData()
model = modeling.HS04Model(cfg)

sampler = data_wrangling.FastSampling(cfg, data)

# Instantiate training data generator
generators = {
    "pho_sem": sampler.sample_generator(x="pho", y="sem"),
    "sem_pho": sampler.sample_generator(x="sem", y="pho"),
    "pho_pho": sampler.sample_generator(x="pho", y="pho"),
    "sem_sem": sampler.sample_generator(x="sem", y="sem"),
    "triangle": sampler.sample_generator(x="ort", y=["pho", "sem"]),
}

# Instantiate optimizer for each task
optimizers = {
    "pho_pho": tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    "sem_sem": tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    "pho_sem": tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    "sem_pho": tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    "triangle": tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
}

# Instantiate loss_fn for each task
loss_fns = {
    "pho_pho": tf.keras.losses.BinaryCrossentropy(),
    "sem_sem": tf.keras.losses.BinaryCrossentropy(),
    "pho_sem": tf.keras.losses.BinaryCrossentropy(),
    "sem_pho": tf.keras.losses.BinaryCrossentropy(),
    "triangle": tf.keras.losses.BinaryCrossentropy(),
}

# Mean loss (for TensorBoard)
train_losses = {
    "pho_pho": tf.keras.metrics.Mean("train_loss_pho_pho", dtype=tf.float32),
    "sem_sem": tf.keras.metrics.Mean("train_loss_sem_sem", dtype=tf.float32),
    "pho_sem": tf.keras.metrics.Mean("train_loss_pho_sem", dtype=tf.float32),
    "sem_pho": tf.keras.metrics.Mean("train_loss_sem_pho", dtype=tf.float32),
    "triangle": tf.keras.metrics.Mean("train_loss_triangle", dtype=tf.float32),
}

# Train metrics
train_acc = {
    "pho_pho": metrics.PhoAccuracy("acc_pho_pho"),
    "sem_sem": metrics.RightSideAccuracy("acc_sem_sem"),
    "pho_sem": metrics.RightSideAccuracy("acc_pho_sem"),
    "sem_pho": metrics.PhoAccuracy("acc_sem_pho"),
    "triangle_pho": metrics.PhoAccuracy("acc_triangle_pho"),
    "triangle_sem": metrics.RightSideAccuracy("acc_triangle_sem"),
}


# %% Train step (Phase 1)


def get_train_step_phase1():
    """Wrap universal train step creator"""

    @tf.function
    def train_step(x, y, model, task, loss_fn, optimizer, train_metric, train_losses):

        train_weights_name = [x + ":0" for x in modeling.WEIGHTS_AND_BIASES[task]]
        train_weights = [x for x in model.weights if x.name in train_weights_name]

        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = loss_fn(y, y_pred)

        grads = tape.gradient(loss_value, train_weights)
        optimizer.apply_gradients(zip(grads, train_weights))

        # Mean loss for Tensorboard
        train_losses.update_state(loss_value)

        # Metric for last time step (output first dimension is time ticks, from -cfg.output_ticks to end)
        train_metric.update_state(tf.cast(y[-1], tf.float32), y_pred[-1])

    return train_step


train_steps = {
    "pho_pho": get_train_step_phase1(),
    "pho_sem": get_train_step_phase1(),
    "sem_sem": get_train_step_phase1(),
    "sem_pho": get_train_step_phase1(),
}


# %% Trainstep (Phase 2)
@tf.function
def train_step_triangle(
    x,
    y,
    model,
    task,
    loss_fn,
    optimizer,
    train_metric_pho,
    train_metric_sem,
    train_losses,
):

    train_weights_name = [x + ":0" for x in modeling.WEIGHTS_AND_BIASES[task]]
    train_weights = [x for x in model.weights if x.name in train_weights_name]

    with tf.GradientTape() as tape:
        pho_pred, sem_pred = model(x, training=True)
        loss_value_pho = loss_fn(y[0], pho_pred)
        loss_value_sem = loss_fn(y[1], sem_pred)
        loss_value = loss_value_pho + loss_value_sem

    grads = tape.gradient(loss_value, train_weights)
    optimizer.apply_gradients(zip(grads, train_weights))

    # Mean loss for Tensorboard
    train_losses.update_state(loss_value)

    # Metric for last time step (output first dimension is time ticks, from -cfg.output_ticks to end)
    train_metric_pho.update_state(tf.cast(y[0][-1], tf.float32), pho_pred[-1])
    train_metric_sem.update_state(tf.cast(y[1][-1], tf.float32), sem_pred[-1])


train_steps["triangle"] = train_step_triangle


# %% Train model

model.build()
phase1_tasks = ["pho_sem", "sem_pho", "pho_pho", "sem_sem"]
phase1_tasks_probability = [0.4, 0.4, 0.1, 0.1]

# TensorBoard writer
train_summary_writer = tf.summary.create_file_writer(cfg.path["tensorboard_folder"])

for epoch in range(cfg.total_number_of_epoch):
    start_time = time.time()

    for step in range(cfg.steps_per_epoch):
        # Intermix tasks (Draw a new task in each step)
        task = np.random.choice(phase1_tasks, p=phase1_tasks_probability)
        model.set_active_task(task)
        x_batch_train, y_batch_train = next(generators[task])

        if task == "triangle":
            train_steps[task](
                x_batch_train,
                y_batch_train,
                model,
                task,
                loss_fns[task],
                optimizers[task],
                train_acc["triangle_pho"],
                train_acc["triangle_sem"],
                train_losses[task],
            )
        else:
            train_steps[task](
                x_batch_train,
                y_batch_train,
                model,
                task,
                loss_fns[task],
                optimizers[task],
                train_acc[task],
                train_losses[task],
            )

    # End of epoch operations

    ## Log all scalar metrics (losses and metrics)and histogram (weights and biases) to tensorboard
    with train_summary_writer.as_default():
        [
            tf.summary.scalar(f"loss_{x}", train_losses[x].result(), step=epoch)
            for x in train_losses.keys()
        ]
        [
            tf.summary.scalar(f"acc_{x}", train_acc[x].result(), step=epoch)
            for x in train_acc.keys()
        ]
        [tf.summary.histogram(f"{x.name}", x, step=epoch) for x in model.weights]

    ## Print status
    compute_time = time.time() - start_time
    print(f"Epoch {epoch + 1} trained for {compute_time:.0f}s")
    print(
        "Losses:",
        [f"{x}: {train_losses[x].result().numpy()}" for x in phase1_tasks],
    )
    clear_output(wait=True)

    ## Save weights
    if (epoch < 10) or ((epoch + 1) % 10 == 0):
        weight_path = cfg.path["weights_checkpoint_fstring"].format(epoch=epoch + 1)
        model.save_weights(weight_path, overwrite=True, save_format="tf")

    ## Reset metric and loss
    [train_losses[x].reset_states() for x in train_losses.keys()]
    [train_acc[x].reset_states() for x in train_acc.keys()]

# End of training ops
# model.save(cfg.path["save_model_folder"])
print("Done")




# %% Universal testset format

class testset():
    """Universal test set object for evaluating model results
    1. Single condition, single metric, single value output for maximum capatibility
    2. Model level info should be stored at separate table, and merge it at the end
    """
    def __init__(self, name, cfg, model, x_test, y_test, metrics):
        self.name = name
        self.cfg = cfg
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.metrics = metrics
    
    def _convert_dict_to_df(self, x):
        df = pd.DataFrame.from_dict({(epoch, timetick): x[epoch][timetick] 
            for epoch in x.keys()
            for timetick in x[epoch].keys()
            }, orient='index')

        df.index.rename(['epoch', 'timeticks'], inplace=True)
        df.reset_index(inplace=True)
        return df

    def eval_all(self, model_id=None, testset_label=None, condition_label=None):
        output = {}
        for epoch in self.cfg.saved_epoches:
            output[epoch] = self._eval_one_epoch(epoch)

        df = self._convert_dict_to_df(output)

        try:
            df["model_id"] = model_id
            df["testset"] = testset_label
            df["condition"] = condition_label
        except:
            pass

        return df
            
    def _eval_one_epoch(self, epoch):
        checkpoint = self.cfg.path["weights_checkpoint_fstring"].format(epoch=epoch)
        self.model.load_weights(checkpoint)
        pred_y = self.model([self.x_test] * self.cfg.n_timesteps)

        output = {}
        if type(pred_y) is list:
            for i, pred_y_at_this_time in enumerate(pred_y):
                tick = self.cfg.n_timesteps - self.cfg.output_ticks + i + 1
                output[tick] = self._eval_one_timetick(pred_y_at_this_time)
        else:
            output[self.cfg.n_timesteps] = self._eval_one_timetick(pred_y)

        return output

    def _eval_one_timetick(self, y_pred):

        output = {}
        for metric in self.metrics:
            metric.update_state(self.y_test, y_pred)
            output[metric.name] = metric.result().numpy()

        return output

# %%
metric_1 = metrics.RightSideAccuracy("right_side_acc")
metric_2 = metrics.PhoAccuracy("acc")

model.set_active_task('sem_pho')
t = testset(name="test",
            cfg=cfg, 
            model=model, 
            x_test=data.testsets["homophone"]["sem"], 
            y_test=data.testsets["homophone"]["pho"],
            metrics=[metric_1, metric_2])

x = t.eval_all(model_id=123, testset_label="homophone", condition_label="homophone")
x
# %%



# %%

df = pd.DataFrame.from_dict({(epoch, timetick): x[epoch][timetick] 
    for epoch in x.keys()
    for timetick in x[epoch].keys()
    }, orient='index')

df.index.rename(['epoch', 'timeticks'], inplace=True)
df.reset_index()



# %% Eval model
data = data_wrangling.MyData()
model = modeling.HS04Model(cfg)
model.build()
model.set_active_task("pho_sem")

# Instantiate metrics
ps_homophone_acc = metrics.RightSideAccuracy("ps_homophone_acc")
ps_non_homophone_acc = metrics.RightSideAccuracy("ps_non_homophone_acc")
ps_train_acc = metrics.RightSideAccuracy("ps_train_acc")


def my_eval(model, cfg, x, y, metrics):
    pred_y = model([x] * cfg.n_timesteps)

    output = []
    for metric in metrics:
        metric.update_state(y, pred_y[-1])
        output.append(metric.result().numpy())

    return output


def eval_oral_phase_ps(checkpoint):

    model.load_weights(checkpoint)

    non_homophone = my_eval(
        model,
        cfg,
        data.testsets["non_homophone"]["pho"],
        data.testsets["non_homophone"]["sem"],
        [ps_non_homophone_acc],
    )

    homophone = my_eval(
        model,
        cfg,
        data.testsets["homophone"]["pho"],
        data.testsets["homophone"]["sem"],
        [ps_homophone_acc],
    )

    all_train = my_eval(
        model,
        cfg,
        data.pho_train,
        data.sem_train,
        [ps_train_acc],
    )

    return non_homophone[0], homophone[0], all_train[0]


# %%

results = []
for chkpt in tqdm(cfg.path["weights_list"]):
    results.append(eval_oral_phase_ps(chkpt))

df = pd.DataFrame(results)


# %%
df.columns = ["non_homophone", "homophone", "total"]
saved_n = len(cfg.path["weights_list"])
df["epoch"] = np.concatenate(
    [np.linspace(1, 10, 10), np.linspace(20, cfg.total_number_of_epoch, saved_n - 10)]
)

df.plot(x="epoch")

# %% [markdown]
# ## SP performance during oral phase

# %%
data = data_wrangling.MyData()
model = modeling.HS04Model(cfg)
model.build()
model.set_active_task("sem_pho")

# Instantiate metrics
sp_homophone_acc = metrics.PhoAccuracy("sp_homophone_acc")
sp_non_homophone_acc = metrics.PhoAccuracy("sp_non_homophone_acc")
sp_train_acc = metrics.PhoAccuracy("sp_train_acc")


def eval_oral_phase_sp(checkpoint):

    model.load_weights(checkpoint)

    non_homophone = my_eval(
        model,
        cfg,
        data.testsets["non_homophone"]["sem"],
        data.testsets["non_homophone"]["pho"],
        [sp_non_homophone_acc],
    )

    homophone = my_eval(
        model,
        cfg,
        data.testsets["homophone"]["sem"],
        data.testsets["homophone"]["pho"],
        [sp_homophone_acc],
    )

    all_train = my_eval(
        model,
        cfg,
        data.sem_train,
        data.pho_train,
        [sp_train_acc],
    )

    return non_homophone[0], homophone[0], all_train[0]


# %%


results = []
for chkpt in tqdm(cfg.path["weights_list"]):
    results.append(eval_oral_phase_sp(chkpt))

df = pd.DataFrame(results)

df.columns = ["non_homophone", "homophone", "total"]
saved_n = len(cfg.path["weights_list"])
df["epoch"] = np.concatenate(
    [np.linspace(1, 10, 10), np.linspace(20, cfg.total_number_of_epoch, saved_n - 10)]
)

df.plot(x="epoch")


# %% Useful commands
# gcloud compute ssh tensorflow-2-4-20210120-000018 --zone us-east4-b -- -L 6006:localhost:6006
# !tensorboard dev upload --logdir tensorboard_log
