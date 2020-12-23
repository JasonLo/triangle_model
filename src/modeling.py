from evaluate import weight
import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import clear_output


class HS04PS(tf.keras.layers.Layer):
    """
    HS04 implementation
    Phase 1 PS
    """
    # Name all weights and biases that in use in each submodel
    # Since there are 4 sets of hidden layer,
    # when refering to hidden units, we need to state the exact layer in this format: h{from}{to}
    # similarly in cleanup units biases: bias_c{from}{to}
    WEIGHTS_AND_BIASES = {}
    WEIGHTS_AND_BIASES["ps"] = ("w_hps_ph", "w_hps_hs", "w_ss", "w_sc", "w_cs", "bias_hps", "bias_s", "bias_css")
    WEIGHTS_AND_BIASES["sp"] = ("w_hsp_sh", "w_hsp_hp", "w_pp", "w_pc", "w_cp", "bias_hsp", "bias_p", "bias_cpp")
    WEIGHTS_AND_BIASES["pp"] = ("w_pc", "w_cp", "bias_p", "bias_cpp")
    WEIGHTS_AND_BIASES["ss"] = ("w_sc", "w_cs", "bias_s", "bias_css")

    def __init__(self, cfg, name="HS04PS", **kwargs):
        super().__init__(**kwargs)

        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

        self.activation = tf.keras.activations.get(self.activation)

    def build(self, input_shape):
        """Build entire Phase 1 model with frozen weights and biases
        turn trainable on later in call() method
        """

        weight_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        # For SP (incl. PP, since PP is nested within SP)
        self.w_hsp_sh = self.add_weight(
            name="w_hsp_sh",
            shape=(input_shape[-1], self.hidden_sp_units),
            initializer=weight_initializer,
            trainable=False,
        )
        self.w_hsp_hp = self.add_weight(
            name="w_hsp_hp",
            shape=(self.hidden_sp_units, self.pho_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_pp = self.add_weight(
            name="w_pp",
            shape=(self.pho_units, self.pho_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_pc = self.add_weight(
            name="w_pc",
            shape=(self.pho_units, self.pho_cleanup_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_cp = self.add_weight(
            name="w_cp",
            shape=(self.pho_cleanup_units, self.pho_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.bias_hsp = self.add_weight(
            shape=(self.hidden_sp_units,),
            name="bias_hsp",
            initializer="zeros",
            trainable=False,
        )

        self.bias_p = self.add_weight(
            shape=(self.pho_units,),
            name="bias_p",
            initializer="zeros",
            trainable=False,
        )

        self.bias_cpp = self.add_weight(
            shape=(self.pho_cleanup_units,),
            name="bias_cpp",
            initializer="zeros",
            trainable=False,
        )

        # For PS (incl. SS, since SS is nested within PS)
        self.w_hps_ph = self.add_weight(
            name="w_hps_ph",
            shape=(input_shape[-1], self.hidden_ps_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_hps_hs = self.add_weight(
            name="w_hps_hs",
            shape=(self.hidden_ps_units, self.sem_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_ss = self.add_weight(
            name="w_ss",
            shape=(self.sem_units, self.sem_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_sc = self.add_weight(
            name="w_sc",
            shape=(self.sem_units, self.sem_cleanup_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.w_cs = self.add_weight(
            name="w_cs",
            shape=(self.sem_cleanup_units, self.sem_units),
            initializer=weight_initializer,
            trainable=False,
        )

        self.bias_hps = self.add_weight(
            shape=(self.hidden_ps_units,),
            name="bias_hps",
            initializer="zeros",
            trainable=False,
        )

        self.bias_s = self.add_weight(
            shape=(self.sem_units,),
            name="bias_s",
            initializer="zeros",
            trainable=False,
        )

        self.bias_css = self.add_weight(
            shape=(self.sem_cleanup_units,),
            name="bias_css",
            initializer="zeros",
            trainable=False,
        )

        self.built = True

    def call(self, inputs, submodel, training=None):
        """
        call submodel when running model()
        """

        if submodel == "ps":
            # Turn on trainable
            for x in self.WEIGHTS_AND_BIASES[submodel]:
                we
            self.submodel_ps(inputs, training)

        elif submodel == "sp":
            self.submodel_sp(inputs, training)

        elif submodel == "pp":
            self.submodel_pp(inputs, training)

        elif submodel == "ss":
            self.submodel_ss(inputs, training)

        else:
            raise KeyError(f"{submodel}: No such submodel.")


    def submodel_ps(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_h_list, input_s_list, input_c_list = [], [], []
        act_h_list, act_s_list, act_c_list = [], [], []

        # Set inputs to 0
        input_h_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_c_list.append(tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_h_list.append(input_h_list[0] + 0.5)
        act_s_list.append(input_s_list[0] + 0.5)
        act_c_list.append(input_c_list[0] + 0.5)

        for t in range(self.n_timesteps):
            # Inject fresh white noise to weights and biases within pho system in each time step while training
            w_ss = K.in_train_phase(
                self._inject_noise(self.w_ss, self.sem_noise_level),
                self.w_ss,
                training=training,
            )
            w_sc = K.in_train_phase(
                self._inject_noise(self.w_sc, self.sem_noise_level),
                self.w_sc,
                training=training,
            )
            w_cs = K.in_train_phase(
                self._inject_noise(self.w_cs, self.sem_noise_level),
                self.w_cs,
                training=training,
            )
            bias_css = K.in_train_phase(
                self._inject_noise(self.bias_css, self.sem_noise_level),
                self.bias_css,
                training=training,
            )
            bias_s = K.in_train_phase(
                self._inject_noise(self.bias_s, self.sem_noise_level),
                self.bias_s,
                training=training,
            )

            ##### Hidden layer #####
            ph = tf.matmul(inputs[:, t, :], self.w_hps_ph)
            h = self.tau * (ph + self.bias_hps) + (1 - self.tau) * input_h_list[t]

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

        return act_s_list[-self.output_ticks :]



    def _inject_noise(self, x, noise_sd):
        """Inject Gaussian noise if noise_sd > 0"""
        if noise_sd > 0:
            noise = K.random_normal(shape=K.shape(x), mean=0.0, stddev=noise_sd)

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


class HS04Phase1PS(tf.keras.Model):
    def __init__(self, cfg):
        super(HS04Phase1PS, self).__init__()
        self.input_xt = tf.keras.layers.RepeatVector(cfg.n_timesteps, name="Input_xt")
        self.hs04ps = HS04PS(cfg)

    def call(self, inputs):
        x = self.input_xt(inputs)
        x = self.hs04ps(x)
        return x


class CustomBCE(tf.keras.losses.Loss):
    """Binarycross entropy loss with variable zero-error-radius"""

    def __init__(self, radius=0.1, name="bce_with_ZER"):
        super().__init__(name=name)
        self.radius = radius

    @staticmethod
    def _constant_to_tensor(x, dtype):
        return tf.python.framework.constant_op.constant(x, dtype=dtype)

    @staticmethod
    def _backtrack_identity(tensor):
        while tensor.op.type == "Identity":
            tensor = tensor.op.inputs[0]
        return tensor

    @staticmethod
    def zer_replace(target, output, zero_error_radius):
        """Replace output by target if value within zero-error-radius"""
        within_zer = tf.math.less_equal(
            tf.math.abs(output - target), tf.constant(zero_error_radius)
        )
        return tf.where(within_zer, target, output)

    def call(self, y_true, y_pred):
        if not isinstance(
            y_pred,
            (tf.python.framework.ops.EagerTensor, tf.python.ops.variables.Variable),
        ):
            y_pred = CustomBCE._backtrack_identity(y_pred)

        # Replace output by target if value within zero error radius
        zer_output = CustomBCE.zer_replace(y_true, y_pred, self.radius)

        # Clip with a tiny constant to avoid zero division
        epsilon_ = CustomBCE._constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        zer_output = tf.python.ops.clip_ops.clip_by_value(
            zer_output, epsilon_, 1.0 - epsilon_
        )

        # Compute cross entropy from probabilities.
        bce = y_true * tf.python.ops.log(zer_output + K.epsilon())
        bce += (1 - y_true) * tf.python.ops.log(1 - zer_output + K.epsilon())
        return -bce


class ModelCheckpoint_custom(tf.keras.callbacks.Callback):
    """
    Modified from original ModelCheckpoint
    Always save first 10 epochs regardless save period
    """

    def __init__(self, filepath_fstring, save_weights_only=False):
        super().__init__()
        self.filepath_fstring = filepath_fstring
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        if (epoch < 10) or (epoch % 10 == 0):
            filepath = self.filepath_fstring.format(epoch=epoch + 1)
            clear_output(wait=True)
            print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
