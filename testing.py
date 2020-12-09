# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Prepare environment
# %% [markdown]
# ## Import libraries

# %%
get_ipython().run_line_magic('load_ext', 'lab_black')
import h5py, pickle, os, sys
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
import os, sys

# Add sys path for batch run
sys.path.append("/home/jupyter/tf")


from src import meta, data_wrangling, modeling, evaluate
from IPython.display import clear_output

meta.gpu_mem_cap(2048)  # Put memory cap to allow parallel runs
meta.check_gpu()

# %% [markdown]
# ## Parameters block for Papermill
# - Instead of using model_cfg directly, this extra step is needed for batch run using Papermill
# - Consider carefully the variable type in each cfg setting (Probably automatically check it later...)
#     - Do not use integer (e.g., 1, 2, 3, 0) in variables that can be float32 (e.g., w_oh_noise, tau...)
#     - Use integer with a dot instead (e.g., 1., 2., 3., 0.) to indicate a float32 value
# - To use attractor, two params must be config, 
#     - 1) embed_attractor_cfg --> json cfg file of the pretrain attractor
#         - e.g.: f'models/Attractor_{cleanup_units:02d}/model_config.json'
#     - 2) embed_attractor_h5 --> h5 file of the exact weight (e.g. ep0500.h5 #epoch or c90.h5 #correct rate)
#         - e.g.: 'c00.h5'

# %%
code_name = "foobar"

# Dataset
sample_name = "experimental"
rng_seed = 53797
use_semantic = False

# Model architechture
input_dim = 119
output_dim = 250
hidden_units = 100
cleanup_units = 50

rnn_activation = "sigmoid"
regularizer_const = None
w_initializer = 0.1
zero_error_radius = 0.1

p_noise = 0.0
tau = 1 / 3
max_unit_time = 4.0
output_ticks = 2

# Pre-Training
pretrain_attractor = False
embed_attractor_cfg = None
embed_attractor_h5 = None

# Training
optimizer = "adam"
n_mil_sample = 0.1
batch_size = 128
learning_rate = 0.005
save_freq = 10

# MISC
show_plots_in_notebook = True
batch_name = "not_a_batch"
batch_unique_setting_string = None

# %% [markdown]
# ## Construct model configuration

# %%
d = {}

# Load global cfg variables into a dictionary
for v in meta.model_cfg.minimal_cfgs:
    d[v] = globals()[v]

for v in meta.model_cfg.aux_cfgs:
    try:
        d[v] = globals()[v]
    except:
        pass

# Construct model_cfg object
cfg = meta.model_cfg(**d)

# %% [markdown]
# # Modeling
# %% [markdown]
# ## Build model

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, concatenate, multiply, RepeatVector
from tensorflow.keras.optimizers import Adam, SGD
from src.metrics import ZERCount, OutputOfZeroTarget, OutputOfOneTarget

tf.random.set_seed(cfg.rng_seed)
data = data_wrangling.MyData()

my_metrics = [
    "BinaryAccuracy",  # Stock metric
    "mse",  # Stock metric
    OutputOfZeroTarget(),
    OutputOfOneTarget(),
    ZERCount(),
]


def build_model(training=True):
    """
    Create Keras model
    Note that:
    For structural things, such as repeat vector, should build within the model
    For Static calculation of input, it is easier to modify, should build within sample generator
    """

    cfg.noise_on() if training else cfg.noise_off()

    input_o = Input(shape=(cfg.input_dim,), name="Input_O")
    input_o_t = RepeatVector(cfg.n_timesteps, name="Input_Ot")(input_o)

    # Construct semantic input
    if cfg.use_semantic == True:
        raw_s_t = Input(shape=(cfg.n_timesteps, cfg.output_dim), name="Plaut_St")

        input_p = Input(shape=(cfg.output_dim,), name="input_P")
        input_p_t = RepeatVector(cfg.n_timesteps, name="Teaching_Pt")(input_p)

        input_s_t = multiply([raw_s_t, input_p_t], name="Input_St")

        combined = concatenate([input_o_t, input_s_t], name="Combined_input")
        rnn_model = modeling.rnn(cfg)(combined)
        model = Model([input_o, raw_s_t, input_p], rnn_model)

    else:
        rnn_model = modeling.rnn(cfg)(input_o_t)
        model = Model(input_o, rnn_model)

    # Select optimizer
    if cfg.optimizer == "adam":
        op = Adam(
            learning_rate=cfg.learning_rate, beta_1=0.0, beta_2=0.999, amsgrad=False
        )

    elif cfg.optimizer == "sgd":
        op = SGD(cfg.learning_rate)

    # Select zero error radius (by chossing custom loss function zer_bce())

    if cfg.zero_error_radius is not None:
        print(f"Using zero-error-radius of {cfg.zero_error_radius}")
        model.compile(
            loss=modeling.CustomBCE(radius=cfg.zero_error_radius),
            optimizer=op,
            metrics=my_metrics,
        )

    elif cfg.zero_error_radius is None:
        print(f"No zero-error-radius")
        model.compile(loss="binary_crossentropy", optimizer=op, metrics=me)

    model.summary()
    return model


model = build_model(training=True)

# %% [markdown]
# ## Arm attractor

# %%
if cfg.pretrain_attractor is True:
    print("Found attractor info in config (cfg), arming attractor...")
    attractor_cfg = meta.model_cfg(cfg.embed_attractor_cfg, bypass_chk=True)
    attractor_obj = modeling.attractor(attractor_cfg, cfg.embed_attractor_h5)
    model = modeling.arm_attractor(model, attractor_obj)
    evaluate.plot_variables(model)
else:
    print("Config indicates no attractor, I have do nothing.")

# %% [markdown]
# ## Training

# %%
# Create sampling instance
my_sampling = data_wrangling.Sampling(cfg, data)

checkpoint = modeling.ModelCheckpoint_custom(
    cfg.path_weights_checkpoint, save_weights_only=True, period=cfg.save_freq,
)

if cfg.batch_name is None:
    tboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"tensorboard_log/{cfg.code_name}", histogram_freq=1
    )
else:
    tboard = tf.keras.callbacks.TensorBoard(
        log_dir=f"tensorboard_log/batch_{cfg.batch_name}/{cfg.code_name}", histogram_freq=1
    )

history = model.fit(
    my_sampling.sample_generator(),
    steps_per_epoch=cfg.steps_per_epoch,
    epochs=cfg.nEpo,
    verbose=0,
    callbacks=[checkpoint, tboard],
)

# Saving history and model
pickle_out = open(cfg.path_history_pickle, "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

clear_output()
print("Training done")

# %% [markdown]
# # results

# %%
# Must turn training mode off before evaluation
model = build_model(training=False)

# %% [markdown]
# ## Parse item level stats

# %%
# Strain
strain = evaluate.strain_eval(cfg, data, model)
strain.start_evaluate(
    test_use_semantic=cfg.use_semantic, output=cfg.path_model_folder + "result_strain_item.csv"
)

# Grain
grain = evaluate.grain_eval(cfg, data, model)
grain.start_evaluate(output=cfg.path_model_folder + "result_grain_item.csv")

# Taraban
taraban = evaluate.taraban_eval(cfg, data, model)
taraban.start_evaluate(
    test_use_semantic=cfg.use_semantic, output=cfg.path_model_folder + "result_taraban_item.csv"
)

# Glushko
glushko = evaluate.glushko_eval(cfg, data, model)
glushko.start_evaluate(test_use_semantic=cfg.use_semantic, output=cfg.path_model_folder + "result_glushko_item.csv")

# %% [markdown]
# ## Create visualization 

# %%
alt.renderers.enable("html")
vis = evaluate.vis(cfg.path_model_folder)

# %% [markdown]
# ### Training history

# %%
vis.training_hist().save(cfg.path_plot_folder + "training_history.html")
if show_plots_in_notebook:
    vis.training_hist().display()

# %% [markdown]
# ### Strain and Grain plots

# %%
sg = vis.plot_dev_interactive("acc", ["strain", "grain"]).properties(
    title="Accuracy in Strain and Grain"
)
sg.save(cfg.path_plot_folder + "development_sg.html")
if show_plots_in_notebook:
    sg.display()

# %% [markdown]
# ### Taraban and Glushko plots

# %%
tb = vis.plot_dev_interactive("acc", ["taraban", "glushko"]).properties(
    title="Accuracy in Taraban and Glushko by condition"
)
tb.save(cfg.path_plot_folder + "development_tg.html")

if show_plots_in_notebook:
    tb.display()
    

# %% [markdown]
# ### Grain plots

# %%
small = vis.plot_dev_interactive("acc_small_grain", exp=["grain"]).properties(
    title="Small Grain Response"
)
large = vis.plot_dev_interactive("acc_large_grain", exp=["grain"]).properties(
    title="Large Grain Response"
)
grain_plot = (small | large).properties(
    title="Accuracy of Grain by response and condition"
)
grain_plot.save(cfg.path_plot_folder + "development_grain_by_response.html")
if show_plots_in_notebook:
    grain_plot.display()

# %% [markdown]
# ### Words vs. Nonwords

# %%
wnw_zr = vis.plot_wnw(["INC_HF"], ["unambiguous"]).properties(
    title="Strain (INC_HF) vs. Grain (UN)"
)

all_taraban_conds = list(vis.cdf.loc[vis.cdf.exp == "taraban", "cond"].unique())
wnw_tg = vis.plot_wnw(all_taraban_conds, ["Exception", "Regular"]).properties(
    title="Taraban (all) vs. Glushko (NW)"
)
wnw_plot = wnw_zr | wnw_tg
wnw_plot.save(cfg.path_plot_folder + "wnw.html")

if show_plots_in_notebook:
    wnw_plot.display()

# %% [markdown]
# ### Model weights and biases

# %%
evaluate.weight(cfg.path_weights_list[-1]).violinplot(
    cfg.path_plot_folder + "weight_violin.png"
)


# %%
evaluate.weight(cfg.path_weights_list[-1]).heatmap(
    cfg.path_plot_folder + "weight_heatmap.png"
)


# %%
# Save file before running nbconvert
# !jupyter nbconvert --output-dir $cfg.path_model_folder --to html_toc OSP_master.ipynb
# !jupyter nbconvert --ExecutePreprocessor.timeout=6000 --to html_toc --execute OSP_master.ipynb --output-dir $cfg.path_model_folder
# !tensorboard --logdir tensor_board_log


