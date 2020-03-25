
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers, regularizers
import tensorflow.keras.backend as K
import numpy as np


class rnn_no_cleanup(Layer):
    # In plaut version (rnn_v1), input are identical in O-->H path over timestep
    # Use keras copy layer seems more efficient
    def __init__(self, cfg, **kwargs):
        super(rnn_no_cleanup, self).__init__(**kwargs)

        self.cfg = cfg

        self.rnn_activation = activations.get(cfg.rnn_activation)
        self.weight_regularizer = regularizers.l2(cfg.regularizer_const)

        self.w_oh = self.add_weight(
            name='w_oh',
            shape=(self.cfg.o_input_dim, self.cfg.hidden_units),
            initializer=self.cfg.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_hp = self.add_weight(
            name='w_hp',
            shape=(self.cfg.hidden_units, self.cfg.pho_units),
            initializer=self.cfg.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_pp = self.add_weight(
            name='w_pp',
            shape=(self.cfg.pho_units, self.cfg.pho_units),
            initializer=self.cfg.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.bias_h = self.add_weight(
            shape=(self.cfg.hidden_units, ),
            name='bias_h',
            initializer='zeros',
            trainable=True
        )

        self.bias_p = self.add_weight(
            shape=(self.cfg.pho_units, ),
            name='bias_p',
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Hack for complying keras.layers.concatenate() format
        # Dimension note: (batch, timestep, input_dim)
        # Spliting input_dim below (index = 2)
        if self.cfg.use_semantic == True:
            o_input, s_input = tf.split(
                inputs, [self.cfg.o_input_dim, self.cfg.pho_units], 2
            )
        else:
            o_input = inputs

        ### Trial level init ###
        self.input_h_list = []
        self.input_p_list = []

        self.act_h_list = []
        self.act_p_list = []

        # Set input to 0
        self.input_h_list.append(
            tf.zeros((1, self.cfg.hidden_units), dtype=tf.float32)
        )
        self.input_p_list.append(
            tf.zeros((1, self.cfg.pho_units), dtype=tf.float32)
        )

        # Set activations to 0.5
        self.act_h_list.append(self.input_h_list[0] + 0.5)
        self.act_p_list.append(self.input_p_list[0] + 0.5)

        for t in range(1, self.cfg.n_timesteps + 1):
            # print(f'Time step = {t}')

            # Inject noise to weights in each time step
            if self.cfg.w_oh_noise != 0:
                w_oh = self.inject_noise(self.w_oh, self.cfg.w_oh_noise)
            else:
                w_oh = self.w_oh

            if self.cfg.w_hp_noise != 0:
                w_hp = self.inject_noise(self.w_hp, self.cfg.w_hp_noise)
            else:
                w_hp = self.w_hp

            if self.cfg.w_pp_noise != 0:
                w_pp = self.inject_noise(self.w_pp, self.cfg.w_pp_noise)
            else:
                w_pp = self.w_pp

            ##### Hidden layer #####
            oh = tf.matmul(o_input[:, t - 1, :], w_oh)
            mem_h = self.input_h_list[t - 1]
            h = self.cfg.tau * (oh + self.bias_h) + (1 - self.cfg.tau) * mem_h

            self.input_h_list.append(h)
            self.act_h_list.append(self.rnn_activation(h))

            ##### Phonology layer #####
            hp = tf.matmul(self.act_h_list[t - 1], w_hp)
            pp = tf.matmul(
                self.act_p_list[t - 1],
                tf.linalg.set_diag(w_pp, tf.zeros(self.cfg.pho_units))
            )  # Zero diagonal lock

            mem_p = self.input_p_list[t - 1]

            p = self.cfg.tau * (hp + pp +
                                self.bias_p) + (1 - self.cfg.tau) * mem_p

            if self.cfg.use_semantic == True:
                p += s_input[:, t - 1, :]  # Inject semantic input

            self.input_p_list.append(p)
            self.act_p_list.append(self.rnn_activation(p))

        return self.act_p_list[1:]

    def inject_noise(self, x, noise_sd):
        noise = K.random_normal(shape=K.shape(x), mean=0., stddev=noise_sd)
        return x + noise

    def compute_output_shape(self):
        return tensor_shape.as_shape([1, cfg.pho_units] + cfg.n_timesteps)

    def get_config(self):
        config = {'custom_cfg': self.cfg, 'name': 'rnn'}
        base_config = super(rnn_pho_task, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rnn_no_cleanup_no_pp(Layer):
    # In plaut version (rnn_v1), input are identical in O-->H path over timestep
    # Use keras copy layer seems more efficient
    def __init__(self, cfg, **kwargs):
        super(rnn_no_cleanup_no_pp, self).__init__(**kwargs)

        self.cfg = cfg

        self.rnn_activation = activations.get(cfg.rnn_activation)
        self.weight_regularizer = regularizers.l2(cfg.regularizer_const)

        self.w_oh = self.add_weight(
            name='w_oh',
            shape=(self.cfg.o_input_dim, self.cfg.hidden_units),
            initializer=self.cfg.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_hp = self.add_weight(
            name='w_hp',
            shape=(self.cfg.hidden_units, self.cfg.pho_units),
            initializer=self.cfg.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_pp = self.add_weight(
            name='w_pp',
            shape=(self.cfg.pho_units, self.cfg.pho_units),
            initializer=self.cfg.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.bias_h = self.add_weight(
            shape=(self.cfg.hidden_units, ),
            name='bias_h',
            initializer='zeros',
            trainable=True
        )

        self.bias_p = self.add_weight(
            shape=(self.cfg.pho_units, ),
            name='bias_p',
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Hack for complying keras.layers.concatenate() format
        # Dimension note: (batch, timestep, input_dim)
        # Spliting input_dim below (index = 2)
        if self.cfg.use_semantic == True:
            o_input, s_input = tf.split(
                inputs, [self.cfg.o_input_dim, self.cfg.pho_units], 2
            )
        else:
            o_input = inputs

        ### Trial level init ###
        self.input_h_list = []
        self.input_p_list = []

        self.act_h_list = []
        self.act_p_list = []

        # Set input to 0
        self.input_h_list.append(
            tf.zeros((1, self.cfg.hidden_units), dtype=tf.float32)
        )
        self.input_p_list.append(
            tf.zeros((1, self.cfg.pho_units), dtype=tf.float32)
        )

        # Set activations to 0.5
        self.act_h_list.append(self.input_h_list[0] + 0.5)
        self.act_p_list.append(self.input_p_list[0] + 0.5)

        for t in range(1, self.cfg.n_timesteps + 1):
            # print(f'Time step = {t}')

            # Inject noise to weights in each time step
            if self.cfg.w_oh_noise != 0:
                w_oh = self.inject_noise(self.w_oh, self.cfg.w_oh_noise)
            else:
                w_oh = self.w_oh

            if self.cfg.w_hp_noise != 0:
                w_hp = self.inject_noise(self.w_hp, self.cfg.w_hp_noise)
            else:
                w_hp = self.w_hp

            if self.cfg.w_pp_noise != 0:
                w_pp = self.inject_noise(self.w_pp, self.cfg.w_pp_noise)
            else:
                w_pp = self.w_pp

            ##### Hidden layer #####
            oh = tf.matmul(o_input[:, t - 1, :], w_oh)
            mem_h = self.input_h_list[t - 1]
            h = self.cfg.tau * (oh + self.bias_h) + (1 - self.cfg.tau) * mem_h

            self.input_h_list.append(h)
            self.act_h_list.append(self.rnn_activation(h))

            ##### Phonology layer #####
            hp = tf.matmul(self.act_h_list[t - 1], w_hp)

            mem_p = self.input_p_list[t - 1]

            p = self.cfg.tau * (hp + self.bias_p) + (1 - self.cfg.tau) * mem_p

            if self.cfg.use_semantic == True:
                p += s_input[:, t - 1, :]  # Inject semantic input

            self.input_p_list.append(p)
            self.act_p_list.append(self.rnn_activation(p))

        return self.act_p_list[1:]

    def inject_noise(self, x, noise_sd):
        noise = K.random_normal(shape=K.shape(x), mean=0., stddev=noise_sd)
        return x + noise

    def compute_output_shape(self):
        return tensor_shape.as_shape([1, cfg.pho_units] + cfg.n_timesteps)

    def get_config(self):
        config = {'custom_cfg': self.cfg, 'name': 'rnn'}
        base_config = super(rnn_pho_task, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
