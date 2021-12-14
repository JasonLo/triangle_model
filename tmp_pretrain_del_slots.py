# %% [markdown]
# # Triangle model
# This interactive notebook runs a triangle model
#

# %% [markdown]
# ## Run parameters
# This block is necessary for running with [papermill](https://papermill.readthedocs.io/en/latest/) in run_papermill.py.

# %%
code_name = "pretrain_trim_pho_sgd_10M"
batch_name = None

# Model configs
ort_units = 119
pho_units = 175
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

# Training configs
pretrain_checkpoint = None
optimizer = "sgd"
batch_size = 1
learning_rate = 10
inject_error_ticks = 4
zero_error_radius = 0.1
loss_ramping = True

# Environment configs
wf_compression = "log"
wf_clip_low = 0
wf_clip_high = 10000
task_names = ["sem_pho", "pho_sem", "pho_pho", "sem_sem"]
tasks_ps = [0.4, 0.4, 0.1, 0.1]
total_sample = 10_000_000

# Misc configs
rng_seed = 2021
save_freq = 50
which_gpu = 2


# %% [markdown]
# ## System environment
# Provision GPU resouses, set random seeds, and load environment variables

# %%
import meta

meta.split_gpu(which_gpu=3)
# IMPORTANT: do not import TensorFlow before this line

import os
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

# Set all seeds
os.environ["PYTHONHASHSEED"] = str(rng_seed)
tf.random.set_seed(rng_seed)
np.random.seed(rng_seed)

# Loads .env file
load_dotenv()


# %% [markdown]
# ## Create run configuration

# %%
# tf_root = os.environ.get("TF_ROOT")
# cfg = meta.Config.from_json(
#     os.path.join(tf_root, "models", code_name, "model_config.json")
# )  # Load from json
cfg = meta.Config.from_dict(**globals())
print(cfg)


# %% [markdown]
# ## Create Experience
# - `Experience()` defines what the model is trained on. It consists of one or more `Stage()`.
# - Each `Stage()` describes what tasks are the model trained with, and how often a task is used during training. It contains one or more `Task()`.
# - Each `Task()` contains how fast the corpus is opened (a set of word that can be sampled), defaults to full open.
# - CAUTION: Due to technical constrain, we cannot save the staging details in `Experience` in a json file, it requires the orginal code to recreate `Experience`.
# - For complex experience, visualize with:
#     - `experience.plot_task_probability()`
#     - `experience.plot_corpus()`

# %%
from environment import Task, Stage, Experience

stages = [
    Stage(
        name="one",
        tasks=[Task(x) for x in cfg.task_names],
        stage_sample=cfg.total_sample,
        task_probability_start=cfg.tasks_ps,
    )
]

experience = Experience(stages)


# %% [markdown]
# ## Create model trainer
# - In tf.keras terminology, `Trainer()` is the compiled model.
# - It includes data, model, optimizer, metrics, and loss function, etc.
# - Since each sub-task has its own states, it will create separate optimizer, metrics, losses in each task.
# - Once instantiate, It will automatically load cfg.pretrain_checkpoint if exists.
# - In tf.keras terminology, `trainer.train()` is `model.fit()`.
# - If trainer.train() try_to_resume argument is True, it will automatically load from unfinished training checkpoint.

# %%
from data_wrangling import (
    load_testset,
)  # TODO: rename to load_dataset or something appropriate?
from training import Trainer, TensorBoardManager

train_data = load_testset("dataset/train.pkl.gz")
trainer = Trainer(cfg=cfg, data=train_data, experience=experience)


# %% [markdown]
# ## Train model
# Restore from latest checkpoint if it exists. However, due to technical limit, Environment() will no longer be completely identical (same parameter, but new rng) after resuming from checkpoint. It will affects resuming ONLY, i.e., the model trained in one single session will be fine.

# %%
tb_manager = TensorBoardManager(
    cfg, trainer.model, trainer.train_metrics, trainer.train_losses
)
trainer.train(tensorboard_manager=tb_manager, try_to_resume=True)

del trainer  # Release memory before running tests

# %% [markdown]
# ## Run tests
# See `benchmarks.py` for a selection of tests.

# %%
import benchmarks

# benchmarks.run_oral_homophone(cfg)
benchmarks.run_oral_eval(cfg, testset="train_r100_trimmed")
# benchmarks.run_read_eval(cfg)
# benchmarks.run_lesion(cfg)
# benchmarks.run_lesion_extra(cfg)


## Full training set test
# import evaluate
# test = evaluate.Test(cfg)
# test.eval_train("triangle", to_bq=True)

# %%
