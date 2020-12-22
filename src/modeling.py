import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import clear_output


class HS04Phase1PS(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.input_o_t = tf.keras.layers.RepeatVector(cfg.n_timesteps, name="Input_Ot")
        self.hs04ps = HS04PS(cfg)

    def call(self, inputs):
        x = self.input_o_t(inputs)
        x = self.hs04ps(x)
        return x



class HS04PS(tf.keras.layers.Layer):
    """
    HS04 implementation
    """

    def __init__(self, cfg, name="HS04PS", **kwargs):
        super().__init__(**kwargs)

        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

        self.activation = tf.keras.activations.get(self.activation)

    def build(self, input_shape):
        # Since there are 4 sets of hidden layer, 
        # when refering to hidden units, we need to state the exact layer in this format: h{from}{to}
        # similarly in cleanup units biases: bias_c{from}{to}

        # Name space in rnn PS
        # w_hps_ph
        # w_hps_hs
        # w_ss
        # w_sc
        # w_cs

        # bias_hps
        # bias_s
        # bias_css

        weight_initializer = tf.random_uniform_initializer(
            minval=-0.1, maxval=0.1)

        """Build weights and biases"""
        self.w_hps_ph = self.add_weight(
            name="w_hps_ph",
            shape=(input_shape[-1], self.hidden_ps_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_hps_hs = self.add_weight(
            name="w_hps_hs",
            shape=(self.hidden_ps_units, self.sem_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_ss = self.add_weight(
            name="w_ss",
            shape=(self.sem_units, self.sem_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_sc = self.add_weight(
            name="w_sc",
            shape=(self.sem_units, self.sem_cleanup_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_cs = self.add_weight(
            name="w_cs",
            shape=(self.sem_cleanup_units, self.sem_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.bias_hps = self.add_weight(
            shape=(self.hidden_ps_units,),
            name="bias_hps",
            initializer="zeros",
            trainable=True,
        )

        self.bias_s = self.add_weight(
            shape=(self.sem_units,),
            name="bias_s",
            initializer="zeros",
            trainable=True,
        )

        self.bias_css = self.add_weight(
            shape=(self.sem_cleanup_units,),
            name="bias_css",
            initializer="zeros",
            trainable=True,
        )

        self.built = True

    def call(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_h_list, input_s_list, input_c_list = [], [], []
        act_h_list, act_s_list, act_c_list = [], [], []

        # Set inputs to 0
        input_h_list.append(
            tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_c_list.append(
            tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_h_list.append(input_h_list[0] + 0.5)
        act_s_list.append(input_s_list[0] + 0.5)
        act_c_list.append(input_c_list[0] + 0.5)

        for t in range(self.n_timesteps):
            # Inject fresh white noise to weights and biases within pho system in each time step while training
            w_ss = K.in_train_phase(
                self._inject_noise(self.w_ss, self.sem_noise_level), self.w_ss, training=training
            )
            w_sc = K.in_train_phase(
                self._inject_noise(self.w_sc, self.sem_noise_level), self.w_sc, training=training
            )
            w_cs = K.in_train_phase(
                self._inject_noise(self.w_cs, self.sem_noise_level), self.w_cs, training=training
            )
            bias_css = K.in_train_phase(
                self._inject_noise(self.bias_css, self.sem_noise_level), self.bias_css, training=training
            )
            bias_s = K.in_train_phase(
                self._inject_noise(self.bias_s, self.sem_noise_level), self.bias_s, training=training
            )

            ##### Hidden layer #####
            ph = tf.matmul(inputs[:, t, :], self.w_hps_ph)
            h = self.tau * (ph + self.bias_hps) + \
                (1 - self.tau) * input_h_list[t]

            ##### Phonology layer #####
            hs = tf.matmul(act_h_list[t], self.w_hps_hs)
            ss = tf.matmul(act_s_list[t], w_ss)
            cs = tf.matmul(act_c_list[t], w_cs)

            s = self.tau * (hs + ss + cs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### Cleanup layer #####
            sc = tf.matmul(act_s_list[t], w_sc)
            c = self.tau * (sc + bias_css) + (1 - self.tau) * input_c_list[t]

            # Record this timestep to list
            input_h_list.append(h)
            input_s_list.append(s)
            input_c_list.append(c)

            act_h_list.append(self.activation(h))
            act_s_list.append(self.activation(s))
            act_c_list.append(self.activation(c))

        return act_s_list[-self.output_ticks:]

    def _inject_noise(self, x, noise_sd):
        """Inject Gaussian noise if noise_sd > 0"""
        if noise_sd > 0:
            noise = K.random_normal(
                shape=K.shape(x), mean=0.0, stddev=noise_sd)

            return x + noise
        else:
            return x

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update(
            {
                "ort_units": self.ort_units,
                "pho_units": self.pho_units,
                "sem_units": self.sem_units,
                "hidden_os_units": self.hidden_os_units,
                "hidden_op_units": self.hidden_op_units,
                "hidden_ps_units": self.hidden_ps_units,
                "hidden_sp_units": self.hidden_sp_units,
                "pho_cleanup_units": self.pho_cleanup_units,
                "sem_cleanup_units": self.sem_cleanup_units,
                "pho_noise_level": self.pho_noise_level,
                "sem_noise_level": self.sem_noise_level,
                "activation": tf.keras.activations.serialize(self.activation),
                "tau": self.tau,
                "n_timesteps": self.n_timesteps,
            }
        )
        return cfg


def _constant_to_tensor(x, dtype):
    return tf.python.framework.constant_op.constant(x, dtype=dtype)


def _backtrack_identity(tensor):
    while tensor.op.type == "Identity":
        tensor = tensor.op.inputs[0]
    return tensor


def zer_replace(target, output, zero_error_radius):
    """Replace output by target if value within zero-error-radius
    """
    within_zer = tf.math.less_equal(tf.math.abs(
        output - target), tf.constant(zero_error_radius))
    return tf.where(within_zer, target, output)





class HS04(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.input_o_t = tf.keras.layers.RepeatVector(cfg.n_timesteps, name="Input_Ot")
        self.rnn = RNN(
            activation=cfg.activation,
            pho_units=cfg.pho_units,
            pho_hidden_units=cfg.pho_hidden_units,
            pho_cleanup_units=cfg.pho_cleanup_units,
            pho_noise_level=cfg.pho_noise_level,
            n_timesteps=cfg.n_timesteps,
            tau=cfg.tau,
            output_ticks=cfg.output_ticks,
        )

    def call(self, inputs):
        x = self.input_o_t(inputs)
        x = self.rnn(x)
        return x


class RNN(tf.keras.layers.Layer):
    """
    HS04 implementation
    """

    def __init__(
        self,
        activation,
        pho_units,
        pho_hidden_units,
        pho_cleanup_units,
        pho_noise_level,
        n_timesteps,
        tau,
        output_ticks,
        name="rnn",
        **kwargs
    ):

        super().__init__(**kwargs)
        self.pho_hidden_units = pho_hidden_units
        self.pho_units = pho_units
        self.pho_cleanup_units = pho_cleanup_units
        self.pho_noise_level = pho_noise_level

        self.tau = tau
        self.n_timesteps = n_timesteps
        self.output_ticks = output_ticks
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):

        weight_initializer = tf.random_uniform_initializer(
            minval=-0.1, maxval=0.1)

        """Build weights and biases"""
        self.w_oh = self.add_weight(
            name="w_oh",
            shape=(input_shape[-1], self.pho_hidden_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_hp = self.add_weight(
            name="w_hp",
            shape=(self.pho_hidden_units, self.pho_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_pp = self.add_weight(
            name="w_pp",
            shape=(self.pho_units, self.pho_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_pc = self.add_weight(
            name="w_pc",
            shape=(self.pho_units, self.pho_cleanup_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_cp = self.add_weight(
            name="w_cp",
            shape=(self.pho_cleanup_units, self.pho_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.bias_h = self.add_weight(
            shape=(self.pho_hidden_units,),
            name="bias_h",
            initializer="zeros",
            trainable=True,
        )

        self.bias_p = self.add_weight(
            shape=(self.pho_units,),
            name="bias_p",
            initializer="zeros",
            trainable=True,
        )

        self.bias_c = self.add_weight(
            shape=(self.pho_cleanup_units,),
            name="bias_c",
            initializer="zeros",
            trainable=True,
        )

        self.built = True

    def call(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_h_list, input_p_list, input_c_list = [], [], []
        act_h_list, act_p_list, act_c_list = [], [], []

        # Set inputs to 0
        input_h_list.append(
            tf.zeros((1, self.pho_hidden_units), dtype=tf.float32))
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_c_list.append(
            tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_h_list.append(input_h_list[0] + 0.5)
        act_p_list.append(input_p_list[0] + 0.5)
        act_c_list.append(input_c_list[0] + 0.5)

        for t in range(self.n_timesteps):
            # Inject fresh white noise to weights and biases within pho system in each time step while training
            w_pp = K.in_train_phase(
                self._inject_noise(self.w_pp, self.pho_noise_level), self.w_pp, training=training
            )
            w_pc = K.in_train_phase(
                self._inject_noise(self.w_pc, self.pho_noise_level), self.w_pc, training=training
            )
            w_cp = K.in_train_phase(
                self._inject_noise(self.w_cp, self.pho_noise_level), self.w_cp, training=training
            )
            bias_c = K.in_train_phase(
                self._inject_noise(self.bias_c, self.pho_noise_level), self.bias_c, training=training
            )
            bias_p = K.in_train_phase(
                self._inject_noise(self.bias_p, self.pho_noise_level), self.bias_p, training=training
            )

            ##### Hidden layer #####
            oh = tf.matmul(inputs[:, t, :], self.w_oh)
            h = self.tau * (oh + self.bias_h) + \
                (1 - self.tau) * input_h_list[t]

            ##### Phonology layer #####
            hp = tf.matmul(act_h_list[t], self.w_hp)
            pp = tf.matmul(act_p_list[t], w_pp)
            cp = tf.matmul(act_c_list[t], w_cp)

            p = self.tau * (hp + pp + cp + bias_p)
            p += (1 - self.tau) * input_p_list[t]

            ##### Cleanup layer #####
            pc = tf.matmul(act_p_list[t], w_pc)
            c = self.tau * (pc + bias_c) + (1 - self.tau) * input_c_list[t]

            # Record this timestep to list
            input_h_list.append(h)
            input_p_list.append(p)
            input_c_list.append(c)

            act_h_list.append(self.activation(h))
            act_p_list.append(self.activation(p))
            act_c_list.append(self.activation(c))

        return act_p_list[-self.output_ticks:]

    def _inject_noise(self, x, noise_sd):
        """Inject Gaussian noise if noise_sd > 0"""
        if noise_sd > 0:
            noise = K.random_normal(
                shape=K.shape(x), mean=0.0, stddev=noise_sd)

            return x + noise
        else:
            return x

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update(
            {
                "pho_hidden_units": self.pho_hidden_units,
                "pho_units": self.pho_units,
                "pho_cleanup_units": self.pho_cleanup_units,
                "pho_noise_level": self.pho_noise_level,
                "tau": self.tau,
                "n_timesteps": self.n_timesteps,
                "output_ticks": self.output_ticks,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return cfg


def _constant_to_tensor(x, dtype):
    return tf.python.framework.constant_op.constant(x, dtype=dtype)


def _backtrack_identity(tensor):
    while tensor.op.type == "Identity":
        tensor = tensor.op.inputs[0]
    return tensor


def zer_replace(target, output, zero_error_radius):
    """Replace output by target if value within zero-error-radius
    """
    within_zer = tf.math.less_equal(tf.math.abs(
        output - target), tf.constant(zero_error_radius))
    return tf.where(within_zer, target, output)


class CustomBCE(tf.keras.losses.Loss):
    """ Binarycross entropy loss with variable zero-error-radius
    """

    def __init__(self, radius=0.1, name="bce_with_ZER"):
        super().__init__(name=name)
        self.radius = radius

    def call(self, y_true, y_pred):
        if not isinstance(y_pred, (tf.python.framework.ops.EagerTensor, tf.python.ops.variables.Variable)):
            y_pred = _backtrack_identity(y_pred)

        # Replace output by target if value within zero error radius
        zer_output = zer_replace(y_true, y_pred, self.radius)

        # Clip with a tiny constant to avoid zero division
        epsilon_ = _constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        zer_output = tf.python.ops.clip_ops.clip_by_value(
            zer_output, epsilon_, 1.0 - epsilon_)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.python.ops.log(zer_output + K.epsilon())
        bce += (1 - y_true) * tf.python.ops.log(1 - zer_output + K.epsilon())
        return -bce


class ModelCheckpoint_custom(tf.keras.callbacks.Callback):
    """
    Modified from original ModelCheckpoint
    Always save first 10 epochs regardless save period
    """

    def __init__(self, filepath, save_weights_only=False, period=1):
        super(ModelCheckpoint_custom, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (((epoch + 1) % self.period == 0) or epoch < 10):
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            clear_output(wait=True)
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
