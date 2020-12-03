# Fixing frekin Glushko evaluator...
# Seems test_set_input() not returning the third item in list
#%% Import libraries

import importlib

import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Layer, RepeatVector, concatenate,
                                     multiply)
from tensorflow.keras.optimizers import SGD, Adam

import data_wrangling
import evaluate
import meta
import modeling
from metrics import OutputOfOneTarget, OutputOfZeroTarget, ZERCount


#%% 
importlib.reload(evaluate)

cfg = meta.model_cfg(json_file='models/booboo/model_config.json')
data = data_wrangling.MyData()
tf.random.set_seed(cfg.rng_seed)

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
        model.compile(loss="binary_crossentropy",
                      optimizer=op, metrics=my_metrics)

    model.summary()
    return model


model = build_model(training=False)
glushko = evaluate.glushko_eval(cfg, data, model)

# Fixing bug in glushko evaluate

glushko.cfg.path_weights_list
epoch = glushko.cfg.saved_epoch_list[18]
weights = glushko.cfg.path_weights_list[18]
glushko.model.load_weights(weights)


test_input = data_wrangling.test_set_input(
    glushko.x_test, glushko.x_test_wf, glushko.x_test_img,
    glushko.y_true_matrix, epoch, glushko.cfg, test_use_semantic=False
)

test_input[0].shape

y_pred_matrix = glushko.model.predict(test_input)

glushko.start_evaluate(False, cfg.path_model_folder +
                       "result_glushko_item.csv")

# %%
