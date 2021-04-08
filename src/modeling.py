import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import clear_output

# Create dictionary for weight & biases related to each task
## Important: Due to model complexity, cannot use trainable flag to turn on/off training during a vanilla training loop
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
WEIGHTS_AND_BIASES["triangle"] = (
    "w_hos_oh",
    "w_hos_hs",
    "bias_hos",
    "w_hop_oh",
    "w_hop_hp",
    "bias_hop",
)


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
        # self.active_task = "triangle" # Cannot set default task, will trigger inf. recursion for some reason

        self.tasks = {
            "pho_sem": self.task_pho_sem,
            "sem_pho": self.task_sem_pho,
            "pho_pho": self.task_pho_pho,
            "sem_sem": self.task_sem_sem,
            "triangle": self.task_triangle,
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
        output_ticks = K.in_train_phase(self.inject_error_ticks, self.output_ticks, training=training)
        return act_s_list[-output_ticks :]

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

            if t < 8:
                # Clamp input and activation to teaching signal
                input_s_list.append((inputs[t] - 0.5) * 100)   # cheap un-sigmoid without inf.
                act_s_list.append(inputs[t])  
                
            else:
                cs = tf.matmul(act_css_list[t], w_cs)
                ss = tf.matmul(act_s_list[t], w_ss)
                s = self.tau * (cs + ss + bias_s)
                s += (1 - self.tau) * input_s_list[t]

                input_s_list.append(s)
                act_s_list.append(self.activation(s))

            # Clean up unit
            sc = tf.matmul(act_s_list[t], w_sc)
            css = self.tau * (sc + bias_css) + (1 - self.tau) * input_css_list[t]
            
            # Record this timestep to list
            input_css_list.append(css)           
            act_css_list.append(self.activation(css))


        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(self.inject_error_ticks, self.output_ticks, training=training)
        return act_s_list[-output_ticks :]

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
        output_ticks = K.in_train_phase(self.inject_error_ticks, self.output_ticks, training=training)
        return act_p_list[-output_ticks :]

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

            if t < 8:
                # Clamp input and activation to teaching signal
                input_p_list.append((inputs[t] - 0.5) * 100)   # cheap un-sigmoid without inf.
                act_p_list.append(inputs[t])  
                
            else:
                cp = tf.matmul(act_cpp_list[t], w_cp)
                pp = tf.matmul(act_p_list[t], w_pp)
                p = self.tau * (cp + pp + bias_p)
                p += (1 - self.tau) * input_p_list[t]

                input_p_list.append(p)
                act_p_list.append(self.activation(p))

            # Clean up unit
            pc = tf.matmul(act_p_list[t], w_pc)
            cpp = self.tau * (pc + bias_cpp) + (1 - self.tau) * input_cpp_list[t]
            
            # Record this timestep to list
            input_cpp_list.append(cpp)           
            act_cpp_list.append(self.activation(cpp))

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(self.inject_error_ticks, self.output_ticks, training=training)
        return act_p_list[-output_ticks :]

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
        output_ticks = K.in_train_phase(self.inject_error_ticks, self.output_ticks, training=training)
        return act_p_list[-output_ticks :], act_s_list[-output_ticks :]

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


