from tensorflow.keras.layers import Layer
import tensorflow as tf

class rnn(Layer):
    # In plaut version (rnn_v1), input are identical in O-->H path over timestep
    # Use keras copy layer seems more efficient
    def __init__(self, cfg, **kwargs):
        from tensorflow.keras import activations, initializers
        import tensorflow.keras.backend as k

        super(rnn, self).__init__(**kwargs)

        self.tau = cfg.tau
        self.n_timesteps = cfg.n_timesteps
        self.cleanup_units = cfg.cleanup_units
        self.hidden_units = cfg.hidden_units
        self.pho_units = cfg.pho_units

        self.w_oh_noise = cfg.w_oh_noise
        self.w_hp_noise = cfg.w_hp_noise
        self.w_pp_noise = cfg.w_pp_noise
        self.w_pc_noise = cfg.w_pc_noise
        self.w_cp_noise = cfg.w_cp_noise
        self.act_p_noise = cfg.act_p_noise

        self.rnn_activation = cfg.rnn_activation
        self.w_initializer = cfg.w_initializer
        self.rnn_activation = activations.get(cfg.rnn_activation)
        self.learning_rate = cfg.learning_rate

    def build(self, input_shape, **kwargs):
        # Create a trainable weight variable for this layer.

        self.w_oh = self.add_weight(name='w_oh',
                                    shape=(input_shape[1], self.hidden_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.w_hp = self.add_weight(name='w_hp',
                                    shape=(self.hidden_units, self.pho_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.w_pp = self.add_weight(name='w_pp',
                                    shape=(self.pho_units, self.pho_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.w_pc = self.add_weight(name='w_pc',
                                    shape=(self.pho_units, self.cleanup_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.w_cp = self.add_weight(name='w_cp',
                                    shape=(self.cleanup_units, self.pho_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.bias_h = self.add_weight(shape=(self.hidden_units, ),
                                      name='bias_h',
                                      initializer='zeros',
                                      trainable=True)

        self.bias_p = self.add_weight(shape=(self.pho_units, ),
                                      name='bias_p',
                                      initializer='zeros',
                                      trainable=True)

        self.bias_c = self.add_weight(shape=(self.cleanup_units, ),
                                      name='bias_c',
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        import tensorflow as tf

        ### Trial level init ###
        self.input_h_list = []
        self.input_p_list = []
        self.input_c_list = []

        self.act_h_list = []
        self.act_p_list = []
        self.act_c_list = []

        # Set input to 0
        self.input_h_list.append(
            tf.zeros((1, self.hidden_units), dtype=tf.float32))
        self.input_p_list.append(
            tf.zeros((1, self.pho_units), dtype=tf.float32))
        self.input_c_list.append(
            tf.zeros((1, self.cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        self.act_h_list.append(self.input_h_list[0] + 0.5)
        self.act_p_list.append(self.input_p_list[0] + 0.5)
        self.act_c_list.append(self.input_c_list[0] + 0.5)

        for t in range(1, self.n_timesteps + 1):
            # print(f'Time step = {t}')

            # Inject noise to weights in each time step
            if self.w_oh_noise != 0:
                noisy_w_oh = self.inject_noise(self.w_oh, self.w_oh_noise)
            if self.w_hp_noise != 0:
                noisy_w_hp = self.inject_noise(self.w_hp, self.w_hp_noise)
            if self.w_pp_noise != 0:
                noisy_w_pp = self.inject_noise(self.w_pp, self.w_pp_noise)
            if self.w_pc_noise != 0:
                noisy_w_pc = self.inject_noise(self.w_pc, self.w_pc_noise)
            if self.w_cp_noise != 0:
                noisy_w_cp = self.inject_noise(self.w_cp, self.w_cp_noise)

            ##### Hidden layer #####
            # Calculate temporary variables for readability
            if self.w_oh_noise != 0:
                oh = tf.matmul(inputs, noisy_w_oh)
            else:
                oh = tf.matmul(inputs, self.w_oh)

            mem_h = self.input_h_list[t - 1]

            h = self.tau * (oh + self.bias_h) + (1 - self.tau) * mem_h

            # Write it to lists which store all input and activation in every time step
            self.input_h_list.append(h)
            self.act_h_list.append(self.rnn_activation(h))

            # ##### Phonology layer #####
            if self.w_hp_noise != 0:
                hp = tf.matmul(self.act_h_list[t - 1], noisy_w_hp)
            else:
                hp = tf.matmul(self.act_h_list[t - 1], self.w_hp)

            if self.w_pp_noise != 0:
                pp = tf.matmul(self.act_p_list[t - 1],
                               tf.linalg.set_diag(
                                   noisy_w_pp, tf.zeros(
                                       self.pho_units)))  # Zero diagonal lock
            else:
                pp = tf.matmul(self.act_p_list[t - 1],
                               tf.linalg.set_diag(
                                   self.w_pp, tf.zeros(
                                       self.pho_units)))  # Zero diagonal lock

            if self.w_cp_noise != 0:
                cp = tf.matmul(self.act_c_list[t - 1], noisy_w_cp)
            else:
                cp = tf.matmul(self.act_c_list[t - 1], self.w_cp)

            mem_p = self.input_p_list[t - 1]
            p = self.tau * (hp + pp + cp + self.bias_p) + (1 -
                                                           self.tau) * mem_p

            self.input_p_list.append(p)

            act_p = self.rnn_activation(p)

            # Inject noise to activation
            if self.act_p_noise != 0:
                act_p = self.inject_noise(act_p, self.act_p_noise)
            self.act_p_list.append(act_p)

            ##### Cleanup layer #####
            if self.w_pc_noise != 0:
                pc = tf.matmul(self.act_p_list[t - 1], noisy_w_pc)
            else:
                pc = tf.matmul(self.act_p_list[t - 1], self.w_pc)

            mem_c = self.input_c_list[t - 1]
            c = self.tau * (pc + self.bias_c) + (1 - self.tau) * mem_c

            self.input_c_list.append(c)
            self.act_c_list.append(self.rnn_activation(c))

        return self.act_p_list[1:]

    def inject_noise(self, x, noise_sd):
        import tensorflow.keras.backend as K
        noise = K.random_normal(shape=K.shape(x), mean=0., stddev=noise_sd)
        return x + noise

    def compute_output_shape(self):
        return [(input_shape[0], self.pho_units), self.n_timesteps]

    def get_config(self):
        config = {'custom_cfg': self.cfg, 'name': 'rnn'}
        base_config = super(rnn_pho_task, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rnn_pho_task(Layer):
    def __init__(self, cfg, **kwargs):
        from tensorflow.keras import activations, initializers
        import tensorflow.keras.backend as k

        super(rnn_pho_task, self).__init__(**kwargs)
        self.cfg = cfg
        self.tau = cfg.tau
        self.n_timesteps = cfg.n_timesteps
        self.cleanup_units = cfg.cleanup_units
        self.hidden_units = cfg.hidden_units
        self.pho_units = cfg.pho_units

        self.rnn_activation = cfg.rnn_activation
        self.w_initializer = cfg.w_initializer
        self.rnn_activation = activations.get(cfg.rnn_activation)
        self.out_activation = activations.get('sigmoid')
        self.learning_rate = cfg.learning_rate

    def build(self, input_shape, **kwargs):
        # Create a trainable weight variable for this layer.

        self.w_oh = self.add_weight(name='w_oh',
                                    shape=(input_shape[1], self.hidden_units),
                                    initializer='zeros',
                                    trainable=False)

        self.w_hp = self.add_weight(name='w_hp',
                                    shape=(self.hidden_units, self.pho_units),
                                    initializer='zeros',
                                    trainable=False)

        self.w_pp = self.add_weight(name='w_pp',
                                    shape=(self.pho_units, self.pho_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.w_pc = self.add_weight(name='w_pc',
                                    shape=(self.pho_units, self.cleanup_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.w_cp = self.add_weight(name='w_cp',
                                    shape=(self.cleanup_units, self.pho_units),
                                    initializer=self.w_initializer,
                                    trainable=True)

        self.bias_h = self.add_weight(shape=(self.hidden_units, ),
                                      name='bias_h',
                                      initializer='zeros',
                                      trainable=False)

        self.bias_p = self.add_weight(shape=(self.pho_units, ),
                                      name='bias_p',
                                      initializer='zeros',
                                      trainable=True)

        self.bias_c = self.add_weight(shape=(self.cleanup_units, ),
                                      name='bias_c',
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        import tensorflow as tf

        ### Trial level init ###
        self.input_p_list = []
        self.input_c_list = []

        self.act_p_list = []
        self.act_c_list = []

        # Set input to 0
        # self.input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        self.input_p_list.append(inputs * 3)
        self.input_c_list.append(
            tf.zeros((1, self.cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        self.act_p_list.append(self.rnn_activation(self.input_p_list[0]))
        self.act_c_list.append(self.input_c_list[0] + 0.5)

        for t in range(1, self.n_timesteps + 1):

            # ##### Phonology layer #####
            pp = tf.matmul(self.act_p_list[t - 1],
                           tf.linalg.set_diag(
                               self.w_pp,
                               tf.zeros(self.pho_units)))  # Zero diagonal lock
            cp = tf.matmul(self.act_c_list[t - 1], self.w_cp)

            mem_p = self.input_p_list[t - 1]
            p = self.tau * (pp + cp + self.bias_p) + (1 - self.tau) * mem_p

            self.input_p_list.append(p)

            if self.n_timesteps <= 14:  # Hard code for now, assuming tau = 0.2, at <=2.8 unit time
                act_p = inputs
            else:
                act_p = self.rnn_activation(p)

            self.act_p_list.append(act_p)

            ##### Cleanup layer #####
            pc = tf.matmul(self.act_p_list[t - 1], self.w_pc)

            mem_c = self.input_c_list[t - 1]
            c = self.tau * (pc + self.bias_c) + (1 - self.tau) * mem_c

            self.input_c_list.append(c)
            self.act_c_list.append(self.rnn_activation(c))

        return self.act_p_list[15:]

    def inject_noise(self, x, noise_sd):
        import tensorflow.keras.backend as K
        noise = K.random_normal(shape=K.shape(x), mean=0., stddev=noise_sd)
        return x + noise

    def compute_output_shape(self):
        return [(input_shape[0], self.pho_units), self.n_timesteps - 14
                ]  # Hard code for now... remaining time steps output

    def get_config(self):
        config = {'custom_cfg': self.cfg, 'name': 'rnn'}
        base_config = super(rnn_pho_task, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
