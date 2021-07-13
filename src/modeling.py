import tensorflow as tf
import tensorflow.keras.backend as K

# Create dictionary for trainable weight & biases related to each task (only affect weight update step)
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
WEIGHTS_AND_BIASES["triangle"] = (
    "w_hos_oh",
    "w_hos_hs",
    "bias_hos",
    "w_hop_oh",
    "w_hop_hp",
    "bias_hop",
    "w_pc",
    "w_cp",
    "w_pp",
    "bias_p",
    "bias_cpp",
    "w_sc",
    "w_cs",
    "w_ss",
    "bias_s",
    "bias_css",
    "w_hps_ph",
    "w_hps_hs",
    "bias_hps",
    "w_hsp_sh",
    "w_hsp_hp",
    "bias_hsp",
)

IN_OUT = {}
IN_OUT["triangle"] = ("ort", ["pho", "sem"])
IN_OUT["pho_pho"] = ("pho", "pho")
IN_OUT["pho_sem"] = ("pho", "sem")
IN_OUT["sem_pho"] = ("sem", "pho")
IN_OUT["sem_sem"] = ("sem", "sem")

IN_OUT["ort_pho"] = ("ort", "pho")
IN_OUT["ort_sem"] = ("ort", "sem")

IN_OUT["exp_osp"] = ("ort", "pho")
IN_OUT["exp_ops"] = ("ort", "sem")

class MyModel(tf.keras.Model):
    """Model object with full output in dictionary format"""

    # Do not use model.predict() 
    # Use model() to predict instead

    def __init__(self, cfg, name="my_model", batch_size_override=None, **kwargs):
        super().__init__(**kwargs)

        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

        # Infered variable need to pass manually
        self.n_timesteps = cfg.n_timesteps 

        if batch_size_override is not None:
            self.batch_size = batch_size_override 

        self.activation = tf.keras.activations.get(self.activation)
        
        # self.active_task = "triangle" # Do not set default task here, 
        # will trigger inf. recursion for some unknown reason

        self.tasks = {
            "pho_sem": self.task_pho_sem,
            "sem_pho": self.task_sem_pho,
            "pho_pho": self.task_pho_pho,
            "sem_sem": self.task_sem_sem,
            "ort_sem": self.task_ort_sem,
            "ort_pho": self.task_ort_pho,
            "triangle": self.task_triangle,
            "exp_osp": self.experimental_task_osp,
            "exp_ops": self.experimental_task_ops
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

        # Storage for recurrent mechanism (TAI)
        
        # Input storage
        self.input_hos = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_hop = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_hps = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_sem = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_css = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_hsp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_pho = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_cpp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)

        # Intermediate input for division of labor
        self.input_hps_hs = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_sem_ss = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_css_cs = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_hos_hs = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_hsp_hp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_pho_pp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_cpp_cp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.input_hop_hp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)

        # Activation storage
        self.hos = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.hop = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.hps = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.css = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.sem = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.hsp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.cpp = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        self.pho = tf.TensorArray(tf.float32, size=self.n_timesteps+1, clear_after_read=False)
        # inputs_name = ("input_hos", "input_hop", "input_hps", "input_s", "input_css", "input_hsp", "input_p", "input_cpp")
        # acts_name = ("hos", "hop", "hps", "css", "sem", "hsp", "cpp", "pho")

        # dol_name = ("dol_sem_os", "dol_sem_cs", "dol_sem_ss", "dol_sem_ps")

        self.built = True

    def set_active_task(self, task):
        """Method for switching task"""
        # print(f"Activate task: {task}")
        self.active_task = task

    def call(self, inputs, training=None):
        """
        call active task when running model()
        inputs: model input
        input dimension: [timestep (input should be identical across timestep), item_in_batch, input_unit]
        return: a dictionary of input and activation depending on task
        """
        return self.tasks[self.active_task](inputs, training)

    def task_pho_sem(self, inputs, training=None):

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

        output_dict = {}
        output_dict["input_hps"] = input_hps_list[-output_ticks:]
        output_dict["input_s"] = input_s_list[-output_ticks:]
        output_dict["input_css"] = input_css_list[-output_ticks:]

        output_dict["hps"] = act_hps_list[-output_ticks:]
        output_dict["sem"] = act_s_list[-output_ticks:]
        output_dict["css"] = act_css_list[-output_ticks:]

        return output_dict

    def task_sem_sem(self, inputs, training=None):
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

        output_dict = {}
        output_dict["input_s"] = input_s_list[-output_ticks:]
        output_dict["input_css"] = input_css_list[-output_ticks:]
        output_dict["sem"] = act_s_list[-output_ticks:]
        output_dict["css"] = act_css_list[-output_ticks:]

        return output_dict

    def task_sem_pho(self, inputs, training=None):
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

        output_dict = {}
        output_dict["input_hsp"] = input_hsp_list[-output_ticks:]
        output_dict["input_p"] = input_p_list[-output_ticks:]
        output_dict["input_cpp"] = input_cpp_list[-output_ticks:]
        output_dict["hsp"] = act_hsp_list[-output_ticks:]
        output_dict["pho"] = act_p_list[-output_ticks:]
        output_dict["cpp"] = act_cpp_list[-output_ticks:]

        return output_dict

    def task_pho_pho(self, inputs, training=None):
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

        output_dict = {}
        output_dict["input_p"] = input_p_list[-output_ticks:]
        output_dict["input_cpp"] = input_cpp_list[-output_ticks:]
        output_dict["pho"] = act_p_list[-output_ticks:]
        output_dict["cpp"] = act_cpp_list[-output_ticks:]

        return output_dict

    def task_ort_sem(self, inputs, training=None):
        # init
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

        output_dict = {}
        output_dict["input_hos"] = input_hos_list[-output_ticks:]
        output_dict["input_s"] = input_s_list[-output_ticks:]
        output_dict["input_css"] = input_css_list[-output_ticks:]
        output_dict["hos"] = act_hos_list[-output_ticks:]
        output_dict["sem"] = act_s_list[-output_ticks:]
        output_dict["css"] = act_css_list[-output_ticks:]
        return output_dict

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

        output_dict = {}
        output_dict["input_hop"] = input_hop_list[-output_ticks:]
        output_dict["input_p"] = input_p_list[-output_ticks:]
        output_dict["input_cpp"] = input_cpp_list[-output_ticks:]
        output_dict["hop"] = act_hop_list[-output_ticks:]
        output_dict["pho"] = act_p_list[-output_ticks:]
        output_dict["cpp"] = act_cpp_list[-output_ticks:]
        return output_dict

    def task_triangle(self, inputs, training=None):
        # init
        # dol_name = ("dol_sem_os", "dol_sem_cs", "dol_sem_ss", "dol_sem_ps")

        # Set inputs to 0
        self.input_hos = self.input_hos.write(0, tf.zeros((self.batch_size, self.hidden_os_units), dtype=tf.float32))
        self.input_hop = self.input_hop.write(0, tf.zeros((self.batch_size, self.hidden_op_units), dtype=tf.float32))
        self.input_sem = self.input_sem.write(0, tf.zeros((self.batch_size, self.sem_units), dtype=tf.float32))
        self.input_pho = self.input_pho.write(0, tf.zeros((self.batch_size, self.pho_units), dtype=tf.float32))
        self.input_hps = self.input_hps.write(0, tf.zeros((self.batch_size, self.hidden_ps_units), dtype=tf.float32))
        self.input_hsp = self.input_hsp.write(0, tf.zeros((self.batch_size, self.hidden_sp_units), dtype=tf.float32))
        self.input_css = self.input_css.write(0, tf.zeros((self.batch_size, self.sem_cleanup_units), dtype=tf.float32))
        self.input_cpp = self.input_cpp.write(0, tf.zeros((self.batch_size, self.pho_cleanup_units), dtype=tf.float32))

        self.input_hps_hs = self.input_hps_hs.write(0, tf.zeros((self.batch_size, self.sem_units), dtype=tf.float32))
        self.input_sem_ss = self.input_sem_ss.write(0, tf.zeros((self.batch_size, self.sem_units), dtype=tf.float32))
        self.input_css_cs = self.input_css_cs.write(0, tf.zeros((self.batch_size, self.sem_units), dtype=tf.float32))
        self.input_hos_hs = self.input_hos_hs.write(0, tf.zeros((self.batch_size, self.sem_units), dtype=tf.float32))
        
        self.input_hsp_hp = self.input_hsp_hp.write(0, tf.zeros((self.batch_size, self.pho_units), dtype=tf.float32))
        self.input_pho_pp = self.input_pho_pp.write(0, tf.zeros((self.batch_size, self.pho_units), dtype=tf.float32))
        self.input_cpp_cp = self.input_cpp_cp.write(0, tf.zeros((self.batch_size, self.pho_units), dtype=tf.float32))
        self.input_hop_hp = self.input_hop_hp.write(0, tf.zeros((self.batch_size, self.pho_units), dtype=tf.float32))

        # Set activations to 0.5
        self.hps = self.hps.write(0, self.input_hps.read(0) + 0.5)
        self.sem = self.sem.write(0, self.input_sem.read(0) + 0.5)
        self.css = self.css.write(0, self.input_css.read(0) + 0.5)
        self.hsp = self.hsp.write(0, self.input_hsp.read(0) + 0.5)
        self.pho = self.pho.write(0, self.input_pho.read(0) + 0.5)
        self.cpp = self.cpp.write(0, self.input_cpp.read(0) + 0.5)
        self.hos = self.hos.write(0, self.input_hos.read(0) + 0.5)
        self.hop = self.hop.write(0, self.input_hop.read(0) + 0.5)

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
            self.input_hos = self.input_hos.write(t+1, self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos) + (1 - self.tau) * self.input_hos.read(t))

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(t+1, self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop) + (1 - self.tau) * self.input_hop.read(t))

            ##### Semantic layer #####
            self.input_hps_hs = self.input_hps_hs.write(t+1, tf.matmul(self.hps.read(t), self.w_hps_hs))
            self.input_sem_ss = self.input_sem_ss.write(t+1, tf.matmul(self.sem.read(t), w_ss))
            self.input_css_cs = self.input_css_cs.write(t+1, tf.matmul(self.css.read(t), w_cs))
            self.input_hos_hs = self.input_hos_hs.write(t+1, tf.matmul(self.hos.read(t), self.w_hos_hs))

            self.input_sem = self.input_sem.write(t+1, self.tau * (self.input_hps_hs.read(t+1) + self.input_sem_ss.read(t+1) + self.input_css_cs.read(t+1) + self.input_hos_hs.read(t+1) + bias_s) + (1 - self.tau) * self.input_sem.read(t))

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(t+1, tf.matmul(self.hsp.read(t), self.w_hsp_hp))
            self.input_pho_pp = self.input_pho_pp.write(t+1, tf.matmul(self.pho.read(t), w_pp))
            self.input_cpp_cp = self.input_cpp_cp.write(t+1, tf.matmul(self.cpp.read(t), w_cp))
            self.input_hop_hp = self.input_hop_hp.write(t+1, tf.matmul(self.hop.read(t), self.w_hop_hp))

            self.input_pho = self.input_pho.write(t+1, self.tau * (self.input_hsp_hp.read(t+1) + self.input_pho_pp.read(t+1) + self.input_cpp_cp.read(t+1) + self.input_hop_hp.read(t+1) + bias_p) + (1 - self.tau) * self.input_pho.read(t))

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(t+1, self.tau * (tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps) + (1 - self.tau) * self.input_hps.read(t))

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(t+1, self.tau * (tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp) + (1 - self.tau) * self.input_hsp.read(t))

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(t+1, self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css) + (1 - self.tau) * self.input_css.read(t))

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(t+1, self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp) + (1 - self.tau) * self.input_cpp.read(t))

            # Update activations
            self.hps = self.hps.write(t+1, self.activation(self.input_hps.read(t+1)))
            self.sem = self.sem.write(t+1, self.activation(self.input_sem.read(t+1)))
            self.css = self.css.write(t+1, self.activation(self.input_css.read(t+1)))

            self.hsp = self.hsp.write(t+1, self.activation(self.input_hsp.read(t+1)))
            self.pho = self.pho.write(t+1, self.activation(self.input_pho.read(t+1)))
            self.cpp = self.cpp.write(t+1, self.activation(self.input_cpp.read(t+1)))

            self.hos = self.hos.write(t+1, self.activation(self.input_hos.read(t+1)))
            self.hop = self.hop.write(t+1, self.activation(self.input_hop.read(t+1)))

            # Testing new format
            # print(f"t={t}, s shape={s.shape}, tensor_shape={self.input_sem[:,t,:].shape}")
            # self.input_sem[:,t,:].assign(s)

        # output different number of time ticks depending on training/testing
        # output_ticks = K.in_train_phase(
        #     self.inject_error_ticks, self.output_ticks, training=training
        # )
        # inputs_name = ("input_hos", "input_hop", "input_hps", "input_sem", "input_css", "input_hsp", "input_pho", "input_cpp")
        # acts_name = ("hos", "hop", "hps", "css", "sem", "hsp", "cpp", "pho")
        output_dict = {}
        output_dict["input_hos"] = self.input_hos.stack()
        output_dict["input_hop"] = self.input_hop.stack()
        output_dict["input_hps"] = self.input_hps.stack()
        output_dict["input_sem"] = self.input_sem.stack()
        output_dict["input_css"] = self.input_css.stack()
        output_dict["input_hsp"] = self.input_hsp.stack()
        output_dict["input_pho"] = self.input_pho.stack()
        output_dict["input_cpp"] = self.input_cpp.stack()

        output_dict["input_hps_hs"] = self.input_hps_hs.stack()
        output_dict["input_sem_ss"] = self.input_sem_ss.stack()
        output_dict["input_css_cs"] = self.input_css_cs.stack()
        output_dict["input_hos_hs"] = self.input_hos_hs.stack()
        output_dict["input_hsp_hp"] = self.input_hsp_hp.stack()
        output_dict["input_pho_pp"] = self.input_pho_pp.stack()
        output_dict["input_cpp_cp"] = self.input_cpp_cp.stack()
        output_dict["input_hop_hp"] = self.input_hop_hp.stack()

        output_dict["hos"] = self.hos.stack()
        output_dict["hop"] = self.hop.stack()
        output_dict["hps"] = self.hps.stack()
        output_dict["css"] = self.css.stack()
        output_dict["sem"] = self.sem.stack()
        output_dict["hsp"] = self.hsp.stack()
        output_dict["cpp"] = self.cpp.stack()
        output_dict["pho"] = self.pho.stack()

        return output_dict

    def experimental_task_osp(self, inputs, training=None):
        """Lesion triangle model with HOP damaged"""
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
            # hop = self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop) [LESION]
            # hop += (1 - self.tau) * input_hop_list[t] [LESION]
 
            ##### Semantic layer #####
            hps_hs = tf.matmul(act_hps_list[t], self.w_hps_hs)
            sem_ss = tf.matmul(act_s_list[t], w_ss)
            css_cs = tf.matmul(act_css_list[t], w_cs)
            hos_hs = tf.matmul(act_hos_list[t], self.w_hos_hs)
            # ort_os = tf.matmul(inputs[t], self.w_os) [No direct path]

            s = self.tau * (hps_hs + sem_ss + css_cs + hos_hs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### Phonology layer #####
            hsp_hp = tf.matmul(act_hsp_list[t], self.w_hsp_hp)
            pho_pp = tf.matmul(act_p_list[t], w_pp)
            cpp_cp = tf.matmul(act_cpp_list[t], w_cp)
            # hop_hp = tf.matmul(act_hop_list[t], self.w_hop_hp) [LESION]
            # ort_op = tf.matmul(inputs[t], self.w_op) [No direct path]

            # p = self.tau * (hsp_hp + pho_pp + cpp_cp + hop_hp + bias_p) [LESION]
            p = self.tau * (hsp_hp + pho_pp + cpp_cp + bias_p) 
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
            # input_hop_list.append(hop) [LESION]

            act_hps_list.append(self.activation(hps))
            act_s_list.append(self.activation(s))
            act_css_list.append(self.activation(css))

            act_hsp_list.append(self.activation(hsp))
            act_p_list.append(self.activation(p))
            act_cpp_list.append(self.activation(cpp))

            act_hos_list.append(self.activation(hos))
            # act_hop_list.append(self.activation(hop)) [LESION]

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )

        output_dict = {}
        output_dict["input_hos"] = input_hos_list[-output_ticks:]
        # output_dict["input_hop"] = input_hop_list[-output_ticks:] [LESION]
        output_dict["input_hps"] = input_hps_list[-output_ticks:]
        output_dict["input_s"] = input_s_list[-output_ticks:]
        output_dict["input_css"] = input_css_list[-output_ticks:]
        output_dict["input_hsp"] = input_hsp_list[-output_ticks:]
        output_dict["input_p"] = input_p_list[-output_ticks:]
        output_dict["input_cpp"] = input_cpp_list[-output_ticks:]

        output_dict["hos"] = act_hos_list[-output_ticks:]
        # output_dict["hop"] = act_hop_list[-output_ticks:] [LESION]
        output_dict["hps"] = act_hps_list[-output_ticks:]
        output_dict["css"] = act_css_list[-output_ticks:]
        output_dict["sem"] = act_s_list[-output_ticks:]
        output_dict["hsp"] = act_hsp_list[-output_ticks:]
        output_dict["cpp"] = act_cpp_list[-output_ticks:]
        output_dict["pho"] = act_p_list[-output_ticks:]

        return output_dict

    def experimental_task_ops(self, inputs, training=None):
        """Lesion triangle model with HOS damaged"""
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
            # hos = self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos) [LESION]
            # hos += (1 - self.tau) * input_hos_list[t] [LESION]

            ##### Hidden layer (OP) #####
            hop = self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
            hop += (1 - self.tau) * input_hop_list[t] 
 
            ##### Semantic layer #####
            hps_hs = tf.matmul(act_hps_list[t], self.w_hps_hs)
            sem_ss = tf.matmul(act_s_list[t], w_ss)
            css_cs = tf.matmul(act_css_list[t], w_cs)
            # hos_hs = tf.matmul(act_hos_list[t], self.w_hos_hs) [LESION]
            # ort_os = tf.matmul(inputs[t], self.w_os) [No direct path]

            # s = self.tau * (hps_hs + sem_ss + css_cs + hos_hs + bias_s) [LESION]
            s = self.tau * (hps_hs + sem_ss + css_cs + bias_s)
            s += (1 - self.tau) * input_s_list[t]

            ##### Phonology layer #####
            hsp_hp = tf.matmul(act_hsp_list[t], self.w_hsp_hp)
            pho_pp = tf.matmul(act_p_list[t], w_pp)
            cpp_cp = tf.matmul(act_cpp_list[t], w_cp)
            hop_hp = tf.matmul(act_hop_list[t], self.w_hop_hp)
            # ort_op = tf.matmul(inputs[t], self.w_op) [No direct path]

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

            # input_hos_list.append(hos) [LESION]
            input_hop_list.append(hop) 

            act_hps_list.append(self.activation(hps))
            act_s_list.append(self.activation(s))
            act_css_list.append(self.activation(css))

            act_hsp_list.append(self.activation(hsp))
            act_p_list.append(self.activation(p))
            act_cpp_list.append(self.activation(cpp))

            # act_hos_list.append(self.activation(hos)) [LESION]
            act_hop_list.append(self.activation(hop)) 

        # output different number of time ticks depending on training/testing
        output_ticks = K.in_train_phase(
            self.inject_error_ticks, self.output_ticks, training=training
        )

        output_dict = {}
        # output_dict["input_hos"] = input_hos_list[-output_ticks:] [LESION]
        output_dict["input_hop"] = input_hop_list[-output_ticks:]
        output_dict["input_hps"] = input_hps_list[-output_ticks:]
        output_dict["input_s"] = input_s_list[-output_ticks:]
        output_dict["input_css"] = input_css_list[-output_ticks:]
        output_dict["input_hsp"] = input_hsp_list[-output_ticks:]
        output_dict["input_p"] = input_p_list[-output_ticks:]
        output_dict["input_cpp"] = input_cpp_list[-output_ticks:]

        # output_dict["hos"] = act_hos_list[-output_ticks:] [LESION]
        output_dict["hop"] = act_hop_list[-output_ticks:] 
        output_dict["hps"] = act_hps_list[-output_ticks:]
        output_dict["css"] = act_css_list[-output_ticks:]
        output_dict["sem"] = act_s_list[-output_ticks:]
        output_dict["hsp"] = act_hsp_list[-output_ticks:]
        output_dict["cpp"] = act_cpp_list[-output_ticks:]
        output_dict["pho"] = act_p_list[-output_ticks:]

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


def get_train_step(task):
    input_name, output_name = IN_OUT[task]

    if task == "triangle":
        
        @tf.function
        def train_step(
            x, y, model, task, loss_fn, optimizer, train_metrics, train_losses
        ):
            """Train a batch, log loss and metrics (last time step only)"""

            train_weights_name = [x + ":0" for x in WEIGHTS_AND_BIASES[task]]
            train_weights = [x for x in model.weights if x.name in train_weights_name]

            
            # TF Automatic differentiation
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                # training flag can be access within model by K.in_train_phase()
                # it can change the behavior in model() (e.g., turn on/off noise)

                loss_value_pho = loss_fn(y['pho'], y_pred['pho'])
                loss_value_sem = loss_fn(y['sem'], y_pred['sem'])
                loss_value = loss_value_pho + loss_value_sem

            grads = tape.gradient(loss_value, train_weights)

            # Weight update
            optimizer.apply_gradients(zip(grads, train_weights))

            # Calculate mean loss and metrics for tensorboard
            # Metrics update (Only last time step)
            for y_name, metrics in train_metrics.items():
                if y_name == "pho":
                    # y[0] is pho, y[0][-1] is last time step in pho
                    [
                        m.update_state(tf.cast(y['pho'][-1], tf.float32), y_pred['pho'][-1])
                        for m in metrics
                    ]
                else:
                    # y[1] is sem, y[0][-1] is last time step in sem
                    [
                        m.update_state(tf.cast(y['sem'][-1], tf.float32), y_pred['sem'][-1])
                        for m in metrics
                    ]

            # Mean loss
            train_losses.update_state(loss_value)

    else:  # Single output tasks

        @tf.function
        def train_step(
            x, y, model, task, loss_fn, optimizer, train_metrics, train_losses
        ):
            train_weights_name = [x + ":0" for x in WEIGHTS_AND_BIASES[task]]
            train_weights = [x for x in model.weights if x.name in train_weights_name]

            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value = loss_fn(y, y_pred[output_name])

            grads = tape.gradient(loss_value, train_weights)
            optimizer.apply_gradients(zip(grads, train_weights))

            [
                m.update_state(tf.cast(y[-1], tf.float32), y_pred[output_name][-1])
                for m in train_metrics
            ]
            train_losses.update_state(loss_value)

    return train_step
