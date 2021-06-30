import tensorflow as tf
import tensorflow.keras.backend as K

# Create dictionary for weight & biases related to each task
## Important: Due to model complexity, it seems that "trainable" flag cannot be use to turn on/off training during a vanilla training loop
## Therefore, it must use custom training loop to control which matrix to perform gradient descent
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
WEIGHTS_AND_BIASES["ort_sem"] = ("w_hos_oh", "w_hos_hs", "w_ss", "bias_hos", "bias_s")
WEIGHTS_AND_BIASES["ort_pho"] = ("w_hop_oh", "w_hop_hp", "w_pp", "bias_hop", "bias_p")
WEIGHTS_AND_BIASES["exp_osp"] = ("w_hos_oh", "w_hos_hs", "bias_hos")
WEIGHTS_AND_BIASES["triangle"] = (
    "w_hos_oh",
    "w_hos_hs",
    "bias_hos",
    "w_hop_oh",
    "w_hop_hp",
    "bias_hop",
    "w_pc", "w_cp", "w_pp", "bias_p", "bias_cpp",
    "w_sc", "w_cs", "w_ss", "bias_s", "bias_css",
    "w_hps_ph", "w_hps_hs", "bias_hps",  
    "w_hsp_sh", "w_hsp_hp", "bias_hsp"
)

IN_OUT = {}
IN_OUT['triangle'] = ('ort', ['pho', 'sem'])
IN_OUT['pho_pho'] = ('pho', 'pho')
IN_OUT['pho_sem'] = ('pho', 'sem')
IN_OUT['sem_pho'] = ('sem', 'pho')
IN_OUT['sem_sem'] = ('sem', 'sem')
IN_OUT['ort_pho'] = ('ort', 'pho')
IN_OUT['ort_sem'] = ('ort', 'sem')



class HS04Model(tf.keras.Model):
    """
    HS04 Phase 1: Oral stage (P/S) pretraining
    HS04 Phase 2: Reading stage (O to P/S simuteneously, freeze all phase 1 matrices)
    Changes to orginal HS04:
    - No direct connection from O to S / P
    """

    def __init__(self, cfg, name="hs04r", **kwargs):
        super().__init__(**kwargs)

        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

        self.activation = tf.keras.activations.get(self.activation)
        # self.active_task = "triangle" # Do not set default task, will trigger inf. recursion for some unknown reason

        self.tasks = {
            "pho_sem": self.task_pho_sem,
            "sem_pho": self.task_sem_pho,
            "pho_pho": self.task_pho_pho,
            "sem_sem": self.task_sem_sem,
            "ort_sem": self.task_ort_sem,
            "ort_pho": self.task_ort_pho,
            "triangle": self.task_triangle,
            "exp_osp": self.experimental_task_osp,
        }

    def build(self, input_shape=None):
        """Build entire model's weights and biases
        Manually control gradient decent in custom training loop
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

        # Phase 2 weight and biases

        # OS branch

        self.w_hos_oh = self.add_weight(
            shape=(self.ort_units, self.hidden_os_units),
            name="w_hos_oh",
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_hos_hs = self.add_weight(
            shape=(self.hidden_os_units, self.sem_units),
            name="w_hos_hs",
            initializer=weight_initializer,
            trainable=True,
        )

        self.bias_hos = self.add_weight(
            shape=(self.hidden_os_units,),
            name="bias_hos",
            initializer="zeros",
            trainable=True,
        )

        # OP branch

        self.w_hop_oh = self.add_weight(
            shape=(self.ort_units, self.hidden_op_units),
            name="w_hop_oh",
            initializer=weight_initializer,
            trainable=True,
        )

        self.w_hop_hp = self.add_weight(
            shape=(self.hidden_op_units, self.pho_units),
            name="w_hop_hp",
            initializer=weight_initializer,
            trainable=True,
        )

        self.bias_hop = self.add_weight(
            shape=(self.hidden_op_units,),
            name="bias_hop",
            initializer="zeros",
            trainable=True,
        )

        self.built = True

    def set_active_task(self, task):
        """Method for switching task"""
        # print(f"Activate task: {task}")
        self.active_task = task

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
        input_hps_list, input_s_list, input_css_list = [], [], []
        act_hps_list, act_s_list, act_css_list = [], [], []

        # Set inputs to 0
        input_hps_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_css_list.append(tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_hps_list.append(input_hps_list[0] + 0.5)
        act_s_list.append(input_s_list[0] + 0.5)
        act_css_list.append(input_css_list[0] + 0.5)

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
            hps = self.tau * (ph + self.bias_hps)
            hps += (1 - self.tau) * input_hps_list[t]

            ##### Semantic layer #####
            hps_hs = tf.matmul(act_hps_list[t], self.w_hps_hs)
            ss = tf.matmul(act_s_list[t], w_ss)
            cs = tf.matmul(act_css_list[t], w_cs)

            s = self.tau * (hps_hs + ss + cs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### S Cleanup layer #####
            sc = tf.matmul(act_s_list[t], w_sc)
            css = self.tau * (sc + bias_css)
            css += (1 - self.tau) * input_css_list[t]

            # Record this timestep to list
            input_hps_list.append(hps)
            input_s_list.append(s)
            input_css_list.append(css)

            act_hps_list.append(self.activation(hps))
            act_s_list.append(self.activation(s))
            act_css_list.append(self.activation(css))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_s_list[-output_ticks:]

    def task_sem_sem(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_s_list, input_css_list = [], []
        act_s_list, act_css_list = [], []

        # Set inputs to 0
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_css_list.append(tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_s_list.append(input_s_list[0] + 0.5)
        act_css_list.append(input_css_list[0] + 0.5)

        for t in range(self.n_timesteps):
            """Inject fresh white noise to weights and biases within pho system
            in each time step while training
            """

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

            ### Semantic ###

            cs = tf.matmul(act_css_list[t], w_cs)
            ss = tf.matmul(act_s_list[t], w_ss)
            s = self.tau * (cs + ss + bias_s)
            #  s = self.tau * (cs + bias_s)
            s += (1 - self.tau) * input_s_list[t]
            input_s_list.append(s)

            if t < 8:
                # Clamp activation to teaching signal
                act_s_list.append(inputs[t])
            else:
                act_s_list.append(self.activation(s))

            ### Cleanup unit ###
            sc = tf.matmul(act_s_list[t], w_sc)
            css = self.tau * (sc + bias_css) + (1 - self.tau) * input_css_list[t]

            # Record this timestep to list
            input_css_list.append(css)
            act_css_list.append(self.activation(css))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_s_list[-output_ticks:]

    def task_sem_pho(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_hsp_list, input_p_list, input_cpp_list = [], [], []
        act_hsp_list, act_p_list, act_cpp_list = [], [], []

        # Set inputs to 0
        input_hsp_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_cpp_list.append(tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_hsp_list.append(input_hsp_list[0] + 0.5)
        act_p_list.append(input_p_list[0] + 0.5)
        act_cpp_list.append(input_cpp_list[0] + 0.5)

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
            hsp = self.tau * (sh + self.bias_hsp)
            hsp += (1 - self.tau) * input_hsp_list[t]

            ##### Phonology layer #####
            hsp_hp = tf.matmul(act_hsp_list[t], self.w_hsp_hp)
            pp = tf.matmul(act_p_list[t], w_pp)
            cp = tf.matmul(act_cpp_list[t], w_cp)

            p = self.tau * (hsp_hp + pp + cp + bias_p)
            p += (1 - self.tau) * input_p_list[t]

            ##### P Cleanup layer #####
            pc = tf.matmul(act_p_list[t], w_pc)
            cpp = self.tau * (pc + bias_cpp) + (1 - self.tau) * input_cpp_list[t]

            # Record this timestep to list
            input_hsp_list.append(hsp)
            input_p_list.append(p)
            input_cpp_list.append(cpp)

            act_hsp_list.append(self.activation(hsp))
            act_p_list.append(self.activation(p))
            act_cpp_list.append(self.activation(cpp))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_p_list[-output_ticks:]

    def task_pho_pho(self, inputs, training=None):
        """
        Dimension note: (batch, timestep, input_dim)
        Hack for complying keras.layers.concatenate() format
        Spliting input_dim below (index = 2)
        """

        # init
        input_p_list, input_cpp_list = [], []
        act_p_list, act_cpp_list = [], []

        # Set inputs to 0
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_cpp_list.append(tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_p_list.append(input_p_list[0] + 0.5)
        act_cpp_list.append(input_cpp_list[0] + 0.5)

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

            # Phonological unit
            cp = tf.matmul(act_cpp_list[t], w_cp)
            pp = tf.matmul(act_p_list[t], w_pp)
            p = self.tau * (cp + pp + bias_p)
            #           p = self.tau * (cp + bias_p)
            p += (1 - self.tau) * input_p_list[t]
            input_p_list.append(p)

            if t < 8:
                # Clamp activation to teaching signal
                act_p_list.append(inputs[t])
            else:
                act_p_list.append(self.activation(p))

            # Clean up unit
            pc = tf.matmul(act_p_list[t], w_pc)
            cpp = self.tau * (pc + bias_cpp) + (1 - self.tau) * input_cpp_list[t]

            # Record this timestep to list
            input_cpp_list.append(cpp)
            act_cpp_list.append(self.activation(cpp))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_p_list[-output_ticks:]

    def task_ort_sem(self, inputs, training=None):

        # init

        # input
        input_hos_list, input_s_list, input_css_list = [], [], []
        act_hos_list, act_s_list, act_css_list = [], [], []

        # Set inputs to 0
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_css_list.append(tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))
        input_hos_list.append(tf.zeros((1, self.hidden_os_units), dtype=tf.float32))

        # Set activations to 0.5
        act_s_list.append(input_s_list[0] + 0.5)
        act_css_list.append(input_css_list[0] + 0.5)
        act_hos_list.append(input_hos_list[0] + 0.5)

        # Recurrent structure over time ticks (Time averaged input)
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

            ##### Hidden layer (OS) #####
            hos = self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
            hos += (1 - self.tau) * input_hos_list[t]

            ##### Semantic layer #####
            sem_ss = tf.matmul(act_s_list[t], w_ss)
            css_cs = tf.matmul(act_css_list[t], w_cs)
            hos_hs = tf.matmul(act_hos_list[t], self.w_hos_hs)

            s = self.tau * (sem_ss + css_cs + hos_hs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### Semantic Cleanup layer #####
            css = self.tau * (tf.matmul(act_s_list[t], w_sc) + bias_css)
            css += (1 - self.tau) * input_css_list[t]

            # Record this timestep to list
            input_s_list.append(s)
            input_css_list.append(css)
            input_hos_list.append(hos)

            act_s_list.append(self.activation(s))
            act_css_list.append(self.activation(css))
            act_hos_list.append(self.activation(hos))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_s_list[-output_ticks:]

    def task_ort_pho(self, inputs, training=None):

        # input
        input_hop_list, input_p_list, input_cpp_list = [], [], []
        act_hop_list, act_p_list, act_cpp_list = [], [], []

        # Set inputs to 0
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_cpp_list.append(tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))
        input_hop_list.append(tf.zeros((1, self.hidden_op_units), dtype=tf.float32))

        # Set activations to 0.5
        act_p_list.append(input_p_list[0] + 0.5)
        act_cpp_list.append(input_cpp_list[0] + 0.5)
        act_hop_list.append(input_hop_list[0] + 0.5)

        # Recurrent structure over time ticks (Time averaged input)
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

            ##### Hidden layer (OP) #####
            hop = self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
            hop += (1 - self.tau) * input_hop_list[t]

            ##### Phonology layer #####
            pho_pp = tf.matmul(act_p_list[t], w_pp)
            cpp_cp = tf.matmul(act_cpp_list[t], w_cp)
            hop_hp = tf.matmul(act_hop_list[t], self.w_hop_hp)

            p = self.tau * (pho_pp + cpp_cp + hop_hp + bias_p)
            p += (1 - self.tau) * input_p_list[t]

            ##### Phonology Cleanup layer #####
            cpp = self.tau * (tf.matmul(act_p_list[t], w_pc) + bias_cpp)
            cpp += (1 - self.tau) * input_cpp_list[t]

            # Record this timestep to list
            input_p_list.append(p)
            input_cpp_list.append(cpp)
            input_hop_list.append(hop)

            act_p_list.append(self.activation(p))
            act_cpp_list.append(self.activation(cpp))
            act_hop_list.append(self.activation(hop))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_p_list[-output_ticks:]

    def task_triangle(self, inputs, training=None):

        # init
        # Ort related
        input_hos_list, input_hop_list = [], []
        act_hos_list, act_hop_list = [], []

        # P-to-S related
        input_hps_list, input_s_list, input_css_list = [], [], []
        act_hps_list, act_s_list, act_css_list = [], [], []

        # S-to-P related
        input_hsp_list, input_p_list, input_cpp_list = [], [], []
        act_hsp_list, act_p_list, act_cpp_list = [], [], []

        # Set inputs to 0
        input_hps_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_css_list.append(tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))

        input_hsp_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_cpp_list.append(tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))

        input_hos_list.append(tf.zeros((1, self.hidden_os_units), dtype=tf.float32))
        input_hop_list.append(tf.zeros((1, self.hidden_op_units), dtype=tf.float32))

        # Set activations to 0.5
        act_hps_list.append(input_hps_list[0] + 0.5)
        act_s_list.append(input_s_list[0] + 0.5)
        act_css_list.append(input_css_list[0] + 0.5)

        act_hsp_list.append(input_hsp_list[0] + 0.5)
        act_p_list.append(input_p_list[0] + 0.5)
        act_cpp_list.append(input_cpp_list[0] + 0.5)

        act_hos_list.append(input_hos_list[0] + 0.5)
        act_hop_list.append(input_hop_list[0] + 0.5)

        # Recurrent structure over time ticks (Time averaged input)
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

            ##### Hidden layer (OS) #####
            hos = self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
            hos += (1 - self.tau) * input_hos_list[t]

            ##### Hidden layer (OP) #####
            hop = self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
            hop += (1 - self.tau) * input_hop_list[t]

            ##### Semantic layer #####
            hps_hs = tf.matmul(act_hps_list[t], self.w_hps_hs)
            sem_ss = tf.matmul(act_s_list[t], w_ss)
            css_cs = tf.matmul(act_css_list[t], w_cs)
            hos_hs = tf.matmul(act_hos_list[t], self.w_hos_hs)
            # ort_os = tf.matmul(inputs[t], self.w_os)

            s = self.tau * (hps_hs + sem_ss + css_cs + hos_hs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### Phonology layer #####
            hsp_hp = tf.matmul(act_hsp_list[t], self.w_hsp_hp)
            pho_pp = tf.matmul(act_p_list[t], w_pp)
            cpp_cp = tf.matmul(act_cpp_list[t], w_cp)
            hop_hp = tf.matmul(act_hop_list[t], self.w_hop_hp)
            # ort_op = tf.matmul(inputs[t], self.w_op)

            p = self.tau * (hsp_hp + pho_pp + cpp_cp + hop_hp + bias_p)
            p += (1 - self.tau) * input_p_list[t]

            ##### Hidden layer (PS) #####
            hps = self.tau * (tf.matmul(act_p_list[t], self.w_hps_ph) + self.bias_hps)
            hps += (1 - self.tau) * input_hps_list[t]

            ##### Hidden layer (SP) #####
            hsp = self.tau * (tf.matmul(act_s_list[t], self.w_hsp_sh) + self.bias_hsp)
            hsp += (1 - self.tau) * input_hsp_list[t]

            ##### Semantic Cleanup layer #####
            css = self.tau * (tf.matmul(act_s_list[t], w_sc) + bias_css)
            css += (1 - self.tau) * input_css_list[t]

            ##### Phonology Cleanup layer #####
            cpp = self.tau * (tf.matmul(act_p_list[t], w_pc) + bias_cpp)
            cpp += (1 - self.tau) * input_cpp_list[t]

            # Record this timestep to list
            input_hps_list.append(hps)
            input_s_list.append(s)
            input_css_list.append(css)

            input_hsp_list.append(hsp)
            input_p_list.append(p)
            input_cpp_list.append(cpp)

            input_hos_list.append(hos)
            input_hop_list.append(hop)

            act_hps_list.append(self.activation(hps))
            act_s_list.append(self.activation(s))
            act_css_list.append(self.activation(css))

            act_hsp_list.append(self.activation(hsp))
            act_p_list.append(self.activation(p))
            act_cpp_list.append(self.activation(cpp))

            act_hos_list.append(self.activation(hos))
            act_hop_list.append(self.activation(hop))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )
        return act_p_list[-output_ticks:], act_s_list[-output_ticks:]

    def experimental_task_osp(self, inputs, training=None):
        """This experimental task is a O to S to P model without any direct connection from O to P.
        The purpose of this task is to isolate wheather SP structure
        """

        # init input and activation stores
        input_hos_list, input_s_list, input_css_list = [], [], []
        input_hsp_list, input_p_list, input_cpp_list = [], [], []

        act_hos_list, act_s_list, act_css_list = [], [], []
        act_hsp_list, act_p_list, act_cpp_list = [], [], []

        # Set inputs to 0
        input_hos_list.append(tf.zeros((1, self.hidden_os_units), dtype=tf.float32))
        input_s_list.append(tf.zeros((1, self.sem_units), dtype=tf.float32))
        input_css_list.append(tf.zeros((1, self.sem_cleanup_units), dtype=tf.float32))
        input_hsp_list.append(tf.zeros((1, self.hidden_ps_units), dtype=tf.float32))
        input_p_list.append(tf.zeros((1, self.pho_units), dtype=tf.float32))
        input_cpp_list.append(tf.zeros((1, self.pho_cleanup_units), dtype=tf.float32))

        # Set activations to 0.5
        act_hos_list.append(input_hos_list[0] + 0.5)
        act_s_list.append(input_s_list[0] + 0.5)
        act_css_list.append(input_css_list[0] + 0.5)
        act_hsp_list.append(input_hsp_list[0] + 0.5)
        act_p_list.append(input_p_list[0] + 0.5)
        act_cpp_list.append(input_cpp_list[0] + 0.5)

        # Division of labor in P
        hsp_hp_list, pho_pp_list, cpp_cp_list, bias_p_list = [], [], [], []

        # Recurrent structure over time ticks (Time averaged input)
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

            ##### Hidden layer (OS) #####
            hos = self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
            hos += (1 - self.tau) * input_hos_list[t]

            ##### Semantic layer #####
            sem_ss = tf.matmul(act_s_list[t], w_ss)
            css_cs = tf.matmul(act_css_list[t], w_cs)
            hos_hs = tf.matmul(act_hos_list[t], self.w_hos_hs)

            s = self.tau * (sem_ss + css_cs + hos_hs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### Phonology layer #####
            hsp_hp = tf.matmul(act_hsp_list[t], self.w_hsp_hp)
            pho_pp = tf.matmul(act_p_list[t], w_pp)
            cpp_cp = tf.matmul(act_cpp_list[t], w_cp)

            # Collect division of labor metrics (Before TAI)
            hsp_hp_list.append(hsp_hp)
            pho_pp_list.append(pho_pp)
            cpp_cp_list.append(cpp_cp)
            bias_p_list.append(bias_p)

            p = self.tau * (hsp_hp + pho_pp + cpp_cp + bias_p)
            p += (1 - self.tau) * input_p_list[t]

            ##### Hidden layer (SP) #####
            hsp = self.tau * (tf.matmul(act_s_list[t], self.w_hsp_sh) + self.bias_hsp)
            hsp += (1 - self.tau) * input_hsp_list[t]

            ##### Semantic Cleanup layer #####
            css = self.tau * (tf.matmul(act_s_list[t], w_sc) + bias_css)
            css += (1 - self.tau) * input_css_list[t]

            ##### Phonology Cleanup layer #####
            cpp = self.tau * (tf.matmul(act_p_list[t], w_pc) + bias_cpp)
            cpp += (1 - self.tau) * input_cpp_list[t]

            # Record this timestep to list
            input_hos_list.append(hos)
            input_s_list.append(s)
            input_css_list.append(css)
            input_hsp_list.append(hsp)
            input_p_list.append(p)
            input_cpp_list.append(cpp)

            act_hos_list.append(self.activation(hos))
            act_s_list.append(self.activation(s))
            act_css_list.append(self.activation(css))
            act_hsp_list.append(self.activation(hsp))
            act_p_list.append(self.activation(p))
            act_cpp_list.append(self.activation(cpp))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )

        # output dictionary with division of labor metrics
        output_dict = dict()
        output_dict["hsp_hp"] = hsp_hp_list[-output_ticks:]
        output_dict["pho_pp"] = pho_pp_list[-output_ticks:]
        output_dict["cpp_cp"] = cpp_cp_list[-output_ticks:]
        output_dict["bias_p"] = bias_p_list[-output_ticks:]
        output_dict["act_s"] = act_s_list[-output_ticks:]
        output_dict["act_p"] = act_p_list[-output_ticks:]

        return output_dict

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


def get_train_step():

    @tf.function
    def train_step(
        x,
        y,
        model,
        task,
        loss_fn,
        optimizer,
        train_metrics,
        train_losses,
    ):

        train_weights_name = [x + ":0" for x in WEIGHTS_AND_BIASES[task]]
        train_weights = [x for x in model.weights if x.name in train_weights_name]

        if task == "triangle":
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value_pho = loss_fn(y[0], y_pred[0])  # Caution order matter
                loss_value_sem = loss_fn(y[1], y_pred[1])
                loss_value = loss_value_pho + loss_value_sem
        else:
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value = loss_fn(y, y_pred)

        grads = tape.gradient(loss_value, train_weights)
        optimizer.apply_gradients(zip(grads, train_weights))

        # Mean loss for Tensorboard
        train_losses.update_state(loss_value)

        # Metric for last time step (output first dimension is time ticks, from -cfg.output_ticks to end) for live results
        if type(train_metrics) is list:
            for i, x in enumerate(train_metrics):
                x.update_state(tf.cast(y[i][-1], tf.float32), y_pred[i][-1])
        else:
            train_metrics.update_state(tf.cast(y[-1], tf.float32), y_pred[-1])

    return train_step