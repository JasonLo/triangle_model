import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import clear_output

# Create dictionary for weight & biases related to each task
## Since there are 4 sets of hidden layers and 2 sets of cleanup units,
## when refering to hidden, we need to state the exact layer in this format: h{from}{to} in weights
## when refering to cleanup, we need to use this format in biases: bias_c{from}{to}
WEIGHTS_AND_BIASES = {}
WEIGHTS_AND_BIASES["pho_sem"] = (
    "w_hps_ph",
    "w_hps_hs",
    "w_ss",
    "w_sc",
    "w_cs",
    "bias_hps",
    "bias_s",
    "bias_css",
)
WEIGHTS_AND_BIASES["sem_pho"] = (
    "w_hsp_sh",
    "w_hsp_hp",
    "w_pp",
    "w_pc",
    "w_cp",
    "bias_hsp",
    "bias_p",
    "bias_cpp",
)
WEIGHTS_AND_BIASES["pho_pho"] = ("w_pc", "w_cp", "bias_p", "bias_cpp")
WEIGHTS_AND_BIASES["sem_sem"] = ("w_sc", "w_cs", "bias_s", "bias_css")


class HS04P1(tf.keras.Model):
    """HS04 Phase 1: Oral stage (P/S) pretraining"""

    def __init__(self, cfg, name="oral", **kwargs):
        super().__init__(**kwargs)

        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

        self.activation = tf.keras.activations.get(self.activation)

        self.tasks = {
            "pho_sem": self.task_pho_sem,
            'sem_pho': self.task_sem_sem,
            'pho_pho': self.task_pho_pho,
            'sem_sem': self.task_sem_sem,
        }

    def build(self, input_shape=None):
        """Build entire Phase 1 model's weights and biases
        Manually control gradient decent in custom training loop
        # CAUTION: indexing is used in
        """

        weight_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)

        # For SP (incl. PP, since PP is nested within SP)
        self.w_hsp_sh = self.add_weight(
            name="w_hsp_sh",
            shape=(self.sem_units, self.hidden_sp_units),
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_hsp_hp = self.add_weight(
            name="w_hsp_hp",
            shape=(self.hidden_sp_units, self.pho_units),
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

        self.bias_hsp = self.add_weight(
            shape=(self.hidden_sp_units,),
            name="bias_hsp",
            initializer="zeros",
            trainable=True,
        )

        self.bias_p = self.add_weight(
            shape=(self.pho_units,),
            name="bias_p",
            initializer="zeros",
            trainable=True,
        )

        self.bias_cpp = self.add_weight(
            shape=(self.pho_cleanup_units,),
            name="bias_cpp",
            initializer="zeros",
            trainable=True,
        )

        # For PS (incl. SS, since SS is nested within PS)
        self.w_hps_ph = self.add_weight(
            name="w_hps_ph",
            shape=(self.pho_units, self.hidden_ps_units),
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

    def set_active_task(self, task):
        # print(f"Activate task: {task}")
        self.active_task = task
        # Turn on trainable
        # Cannot turn on individual weight, handle it in custom training loop

    def call(self, inputs, training=None):
        """
        call active task when running model()
        inputs: model input
        return: prediction
        """
        return self.tasks[self.active_task](inputs, training)

    def task_pho_sem(self, inputs, training=None):
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
            ph = tf.matmul(inputs[t], self.w_hps_ph)
            h = self.tau * (ph + self.bias_hps)
            h += (1 - self.tau) * input_h_list[t]

            ##### Semantic layer #####
            hs = tf.matmul(act_h_list[t], self.w_hps_hs)
            ss = tf.matmul(act_s_list[t], w_ss)
            cs = tf.matmul(act_c_list[t], w_cs)

            s = self.tau * (hs + ss + cs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### S Cleanup layer #####
            sc = tf.matmul(act_s_list[t], w_sc)
            c = self.tau * (sc + bias_css) 
            c += (1 - self.tau) * input_c_list[t]

            # Record this timestep to list
            input_h_list.append(h)
            input_s_list.append(s)
            input_c_list.append(c)

            act_h_list.append(self.activation(h))
            act_s_list.append(self.activation(s))
            act_c_list.append(self.activation(c))

        return act_s_list[-self.output_ticks :]

    def task_sem_pho(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_h_list, input_p_list, input_c_list = [], [], []
        act_h_list, act_p_list, act_c_list = [], [], []

        # Set inputs to 0
        input_h_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_c_list.append(tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_h_list.append(input_h_list[0] + 0.5)
        act_p_list.append(input_p_list[0] + 0.5)
        act_c_list.append(input_c_list[0] + 0.5)

        for t in range(self.n_timesteps):
            # Inject fresh white noise to weights and biases within pho system in each time step while training
            w_pp = K.in_train_phase(
                self._inject_noise(self.w_pp, self.pho_noise_level),
                self.w_pp,
                training=training,
            )
            w_pc = K.in_train_phase(
                self._inject_noise(self.w_pc, self.pho_noise_level),
                self.w_pc,
                training=training,
            )
            w_cp = K.in_train_phase(
                self._inject_noise(self.w_cp, self.pho_noise_level),
                self.w_cp,
                training=training,
            )
            bias_cpp = K.in_train_phase(
                self._inject_noise(self.bias_cpp, self.pho_noise_level),
                self.bias_cpp,
                training=training,
            )
            bias_p = K.in_train_phase(
                self._inject_noise(self.bias_p, self.pho_noise_level),
                self.bias_p,
                training=training,
            )

            ##### Hidden layer #####
            sh = tf.matmul(inputs[t], self.w_hsp_sh)
            h = self.tau * (sh + self.bias_hsp)
            h += (1 - self.tau) * input_h_list[t]

            ##### Phonology layer #####
            hp = tf.matmul(act_h_list[t], self.w_hsp_hp)
            pp = tf.matmul(act_p_list[t], w_pp)
            cp = tf.matmul(act_c_list[t], w_cp)

            p = self.tau * (hp + pp + cp + bias_p)
            p += (1 - self.tau) * input_p_list[t]

            ##### P Cleanup layer #####
            pc = tf.matmul(act_p_list[t], w_pc)
            c = self.tau * (pc + bias_cpp) + (1 - self.tau) * input_c_list[t]

            # Record this timestep to list
            input_h_list.append(h)
            input_p_list.append(p)
            input_c_list.append(c)

            act_h_list.append(self.activation(h))
            act_p_list.append(self.activation(p))
            act_c_list.append(self.activation(c))

        return act_p_list[-self.output_ticks :]



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
