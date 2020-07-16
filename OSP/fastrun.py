# pylint: disable=no-member
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Layer, Input, concatenate, multiply, RepeatVector
from tensorflow.keras import Model
import h5py
import pickle
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import meta
import data_wrangling
import modeling
import evaluate
from IPython.display import clear_output

code_name = "boo_local_zer2"

sample_name = "jay"
rng_seed = 53797
use_semantic = False

# Model architechture
input_dim = 119
output_dim = 250
hidden_units = 100
cleanup_units = 20

pretrain_attractor = False
rnn_activation = "sigmoid"
p_noise = 0.0
tau = 1 / 3
max_unit_time = 4.0
output_ticks = 2

optimizer = "adam"
regularizer_const = None
w_initializer = 0.1
zero_error_radius = 0.2

n_mil_sample = 1.0
batch_size = 128
learning_rate = 0.05
save_freq = 10

bq_dataset = None
batch_unique_setting_string = None

# Construct model configuration
d = {}
for v in meta.model_cfg.minimal_cfgs:
    d[v] = globals()[v]

for v in meta.model_cfg.aux_cfgs:
    try:
        d[v] = globals()[v]
    except:
        pass

cfg = meta.model_cfg(**d)

tf.random.set_seed(cfg.rng_seed)
data = data_wrangling.my_data(cfg)


# Building model


def build_model(training=True):
    """
    Create Keras model
    Note that:
    For structural things, such as repeat vector, should build within the model
    For Static calculation of input, it is easier to modify, should build within sample generator
    """

    cfg.noise_on() if training is True else cfg.noise_off()
    input_o = Input(shape=(cfg.input_dim,), name="Input_O")
    input_o_t = RepeatVector(cfg.n_timesteps, name="Input_Ot")(input_o)

    # Construct semantic input
    if cfg.use_semantic == True:
        raw_s_t = Input(
            shape=(cfg.n_timesteps, cfg.output_dim), name="Plaut_St")

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
            learning_rate=cfg.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False
        )

    elif cfg.optimizer == "sgd":
        op = SGD(cfg.learning_rate)

    # Select zero error radius (by chossing custom loss function zer_bce())
    if cfg.zero_error_radius is not None:
        print(f"Using zero-error-radius of {zero_error_radius}")
        model.compile(
            loss=modeling.CustomBCE(radius=cfg.zero_error_radius),
            optimizer=op,
            metrics=["BinaryAccuracy", "mse"],
        )

    elif cfg.zero_error_radius is None:
        print(f"No zero-error-radius")
        model.compile(
            loss="binary_crossentropy", optimizer=op, metrics=["BinaryAccuracy", "mse"]
        )

    model.summary()
    return model


model = build_model(training=True)


if cfg.pretrain_attractor is True:
    print("Found attractor info in config (cfg), arming attractor...")
    attractor_cfg = meta.model_cfg(cfg.embed_attractor_cfg, bypass_chk=True)
    attractor_obj = modeling.attractor(attractor_cfg, cfg.embed_attractor_h5)
    model = modeling.arm_attractor(model, attractor_obj)
    evaluate.plot_variables(model)
else:
    print("Config indicates no attractor, I have do nothing.")


checkpoint = modeling.ModelCheckpoint_custom(
    cfg.path_weights_checkpoint, save_weights_only=True, period=cfg.save_freq,
)

history = model.fit(
    data_wrangling.sample_generator(cfg, data),
    steps_per_epoch=cfg.steps_per_epoch,
    epochs=cfg.nEpo,
    verbose=0,
    callbacks=[checkpoint],
)

# Save history and model
pickle_out = open(cfg.path_history_pickle, "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

print("Training done")

# Evaluation
model = build_model(training=False)

strain = evaluate.strain_eval(cfg, data, model)
strain.start_evaluate(
    test_use_semantic=False, output=cfg.path_model_folder + "result_strain_item.csv"
)

grain = evaluate.grain_eval(cfg, data, model)
grain.start_evaluate(output=cfg.path_model_folder + "result_grain_item.csv")
