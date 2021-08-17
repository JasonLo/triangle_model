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

IN_OUT["exp_os"] = ("ort", "sem")
IN_OUT["exp_op"] = ("ort", "pho")

WEIGHTS_AND_BIASES["exp_os_ff"] = ("w_hos_oh", "w_hos_hs", "bias_hos", "bias_s")
IN_OUT["exp_os_ff"] = ("ort", "sem")


class MyModel(tf.keras.Model):
    """Model object with full output in dictionary format"""

    # Do not use model.predict(x)
    # Use model(x) to predict instead

    INPUT_ARRAY_NAMES = (
        "input_hos",
        "input_hop",
        "input_hps",
        "input_hsp",
        "input_css",
        "input_cpp",
        "input_sem",
        "input_pho",
        "input_hps_hs",
        "input_sem_ss",
        "input_css_cs",
        "input_hos_hs",
        "input_hsp_hp",
        "input_pho_pp",
        "input_cpp_cp",
        "input_hop_hp",
    )

    ACTIVATION_ARRAY_NAMES = ("hos", "hop", "hps", "hsp", "css", "cpp", "sem", "pho")

    ALL_ARRAY_NAMES = INPUT_ARRAY_NAMES + ACTIVATION_ARRAY_NAMES

    def __init__(self, cfg, name="my_model", batch_size_override=None, **kwargs):
        super().__init__(**kwargs)

        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

        # Inferred variable need to pass manually
        self.n_timesteps = cfg.n_timesteps
        self.activation = tf.keras.activations.get(self.activation)

        # self.active_task = "triangle" # Do not set default task here,
        # will trigger inf. recursion for some unknown reason

        # Explicitly create tasks dictionary for safety
        self.tasks = {
            "pho_sem": self.task_pho_sem,
            "sem_pho": self.task_sem_pho,
            "pho_pho": self.task_pho_pho,
            "sem_sem": self.task_sem_sem,
            "ort_sem": self.task_ort_sem,
            "ort_pho": self.task_ort_pho,
            "triangle": self.task_triangle,
            "exp_osp": self.experimental_task_osp,
            "exp_ops": self.experimental_task_ops,
            "exp_os": self.experimental_task_os,
            "exp_op": self.experimental_task_op,
            "exp_os_ff": self.task_ort_sem_ff,
        }

        # Instead of relying boardcasting, we specifiy the batch size manually for easier debugging
        self.shapes = self._create_shape_map(batch_size=batch_size_override)

    def _create_shape_map(self, batch_size:int=None) -> dict:

        if batch_size is None:
            batch_size = self.batch_size
            
        INPUT_SHAPES = {}
        INPUT_SHAPES["input_hos"] = (batch_size, self.hidden_os_units)
        INPUT_SHAPES["input_hop"] = (batch_size, self.hidden_op_units)
        INPUT_SHAPES["input_sem"] = (batch_size, self.sem_units)
        INPUT_SHAPES["input_pho"] = (batch_size, self.pho_units)
        INPUT_SHAPES["input_hps"] = (batch_size, self.hidden_ps_units)
        INPUT_SHAPES["input_hsp"] = (batch_size, self.hidden_sp_units)
        INPUT_SHAPES["input_css"] = (batch_size, self.sem_cleanup_units)
        INPUT_SHAPES["input_cpp"] = (batch_size, self.pho_cleanup_units)
        INPUT_SHAPES["input_hps_hs"] = (batch_size, self.sem_units)
        INPUT_SHAPES["input_sem_ss"] = (batch_size, self.sem_units)
        INPUT_SHAPES["input_css_cs"] = (batch_size, self.sem_units)
        INPUT_SHAPES["input_hos_hs"] = (batch_size, self.sem_units)
        INPUT_SHAPES["input_hsp_hp"] = (batch_size, self.pho_units)
        INPUT_SHAPES["input_pho_pp"] = (batch_size, self.pho_units)
        INPUT_SHAPES["input_cpp_cp"] = (batch_size, self.pho_units)
        INPUT_SHAPES["input_hop_hp"] = (batch_size, self.pho_units)

        return INPUT_SHAPES

    def build(self, input_shape=None):
        """Build entire model's weights and biases
        Manually control gradient decent in custom training loop
        """

        weight_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        bias_initializer = tf.constant_initializer(value=-5.0)

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
            initializer=bias_initializer,
            trainable=True,
        )

        self.bias_p = self.add_weight(
            shape=(self.pho_units,),
            name="bias_p",
            initializer=bias_initializer,
            trainable=True,
        )

        self.bias_cpp = self.add_weight(
            shape=(self.pho_cleanup_units,),
            name="bias_cpp",
            initializer=bias_initializer,
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
            initializer=bias_initializer,
            trainable=True,
        )

        self.bias_s = self.add_weight(
            shape=(self.sem_units,),
            name="bias_s",
            initializer=bias_initializer,
            trainable=True,
        )

        self.bias_css = self.add_weight(
            shape=(self.sem_cleanup_units,),
            name="bias_css",
            initializer=bias_initializer,
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
            initializer=bias_initializer,
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
            initializer=bias_initializer,
            trainable=True,
        )

        self.built = True

    def set_active_task(self, task):
        """Method for switching task"""
        self.active_task = task

    def call(self, inputs, training=None) -> dict:
        """Call active task when running model().
        
        Args:
            inputs: model input
            input dimension: [timestep (input should be identical across timestep), item_in_batch, input_unit]
        
        return: a dictionary of input and activation depending on task
        """

        # getattr(self, f"task_{self.active_task}")(inputs, training)
        return self.tasks[self.active_task](inputs, training)

    def task_pho_sem(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hps_ph) + self.bias_hps)
                + (1 - self.tau) * self.input_hps.read(t),
            )

            ##### Semantic layer #####

            self.input_hps_hs = self.input_hps_hs.write(
                t + 1, tf.matmul(self.hps.read(t), self.w_hps_hs)
            )
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_hps_hs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_css_cs.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            # Record this timestep to list
            [self._update_activations(t + 1, x) for x in ("hps", "sem", "css")]

        return self._package_output(training=training)

    def task_sem_sem(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ### Semantic ###
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_css_cs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            if t < 8:
                # Clamp activation to teaching signal
                self.sem = self.sem.write(t + 1, inputs[t])
            else:
                self.sem = self.sem.write(
                    t + 1, self.activation(self.input_sem.read(t + 1))
                )

            self._update_activations(t + 1, "css")

        return self._package_output(training=training)

    def task_sem_pho(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hsp_sh) + self.bias_hsp)
                + (1 - self.tau) * self.input_hsp.read(t),
            )

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(
                t + 1, tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            )
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_hsp_hp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_cpp_cp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            [self._update_activations(t + 1, x) for x in ("hsp", "pho", "cpp")]

        return self._package_output(training=training)

    def task_pho_pho(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            # Phonological unit
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_cpp_cp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            if t < 8:
                # Clamp activation to teaching signal
                self.pho = self.pho.write(t + 1, inputs[t])
            else:
                self.pho = self.pho.write(
                    t + 1, self.activation(self.input_pho.read(t + 1))
                )

            self._update_activations(t + 1, "cpp")

        return self._package_output(training=training)

    def task_ort_sem(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Semantic layer #####
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_css_cs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_hos_hs.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            # Update activation
            [self._update_activations(t + 1, x) for x in ("sem", "css", "hos")]

        return self._package_output(training=training)

    def task_ort_sem_ff(self, inputs, training=None):
        """OS feedforward task"""

        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)
            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Semantic layer #####
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau * (self.input_hos_hs.read(t + 1) + bias_s)
                + (1 - self.tau) * self.input_sem.read(t),
            )

            # Update activation
            [self._update_activations(t + 1, x) for x in ("sem", "hos")]

        return self._package_output(training=training)

    def task_ort_pho(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
                + (1 - self.tau) * self.input_hop.read(t),
            )

            ##### Phonology layer #####
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_hop_hp = self.input_hop_hp.write(
                t + 1, tf.matmul(self.hop.read(t), self.w_hop_hp)
            )

            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_cpp_cp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_hop_hp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            [self._update_activations(t + 1, x) for x in ("pho", "cpp", "hop")]

        return self._package_output(training=training)

    def task_triangle(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
                + (1 - self.tau) * self.input_hop.read(t),
            )

            ##### Semantic layer #####
            self.input_hps_hs = self.input_hps_hs.write(
                t + 1, tf.matmul(self.hps.read(t), self.w_hps_hs)
            )
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_hps_hs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_css_cs.read(t + 1)
                    + self.input_hos_hs.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(
                t + 1, tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            )
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_hop_hp = self.input_hop_hp.write(
                t + 1, tf.matmul(self.hop.read(t), self.w_hop_hp)
            )

            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_hsp_hp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_cpp_cp.read(t + 1)
                    + self.input_hop_hp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps)
                + (1 - self.tau) * self.input_hps.read(t),
            )

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp)
                + (1 - self.tau) * self.input_hsp.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def experimental_task_ops(self, inputs, training=None):
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
                + (1 - self.tau) * self.input_hop.read(t),
            )

            ##### Semantic layer #####
            self.input_hps_hs = self.input_hps_hs.write(
                t + 1, tf.matmul(self.hps.read(t), self.w_hps_hs)
            )
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_hps_hs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_css_cs.read(t + 1)
                    # + self.input_hos_hs.read(t + 1) [LESION]
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(
                t + 1, tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            )
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_hop_hp = self.input_hop_hp.write(
                t + 1, tf.matmul(self.hop.read(t), self.w_hop_hp)
            )

            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_hsp_hp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_cpp_cp.read(t + 1)
                    + self.input_hop_hp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps)
                + (1 - self.tau) * self.input_hps.read(t),
            )

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp)
                + (1 - self.tau) * self.input_hsp.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def experimental_task_osp(self, inputs, training=None):
        """Lesion triangle model with HOP damaged"""

        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
                + (1 - self.tau) * self.input_hop.read(t),
            )

            ##### Semantic layer #####
            self.input_hps_hs = self.input_hps_hs.write(
                t + 1, tf.matmul(self.hps.read(t), self.w_hps_hs)
            )
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_hps_hs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_css_cs.read(t + 1)
                    + self.input_hos_hs.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(
                t + 1, tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            )
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )

            self.input_hop_hp = self.input_hop_hp.write(
                t + 1, tf.matmul(self.hop.read(t), self.w_hop_hp)
            )

            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_hsp_hp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_cpp_cp.read(t + 1)
                    # + self.input_hop_hp.read(t + 1) [LESION]
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps)
                + (1 - self.tau) * self.input_hps.read(t),
            )

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp)
                + (1 - self.tau) * self.input_hsp.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def experimental_task_os(self, inputs, training=None):
        """Lesion triangle model with OPS damaged
        Get S output only, should be indentical to OS model
        Extremely slow, please use OS if pass consistency checking
        """
        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
                + (1 - self.tau) * self.input_hop.read(t),
            )

            ##### Semantic layer #####
            self.input_hps_hs = self.input_hps_hs.write(
                t + 1, tf.matmul(self.hps.read(t), self.w_hps_hs)
            )
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    # self.input_hps_hs.read(t + 1) [LESION]
                    self.input_css_cs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_hos_hs.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(
                t + 1, tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            )
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_hop_hp = self.input_hop_hp.write(
                t + 1, tf.matmul(self.hop.read(t), self.w_hop_hp)
            )

            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_hsp_hp.read(t + 1)
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_cpp_cp.read(t + 1)
                    + self.input_hop_hp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps)
                + (1 - self.tau) * self.input_hps.read(t),
            )

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp)
                + (1 - self.tau) * self.input_hsp.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def experimental_task_op(self, inputs, training=None):

        self._init_all_tensor_arrays()

        # Recurrent structure over time ticks (Time averaged input)
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pp, w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_ss, w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            self.input_hos = self.input_hos.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos)
                + (1 - self.tau) * self.input_hos.read(t),
            )

            ##### Hidden layer (OP) #####
            self.input_hop = self.input_hop.write(
                t + 1,
                self.tau * (tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop)
                + (1 - self.tau) * self.input_hop.read(t),
            )

            ##### Semantic layer #####
            self.input_hps_hs = self.input_hps_hs.write(
                t + 1, tf.matmul(self.hps.read(t), self.w_hps_hs)
            )
            self.input_sem_ss = self.input_sem_ss.write(
                t + 1, tf.matmul(self.sem.read(t), w_ss)
            )
            self.input_css_cs = self.input_css_cs.write(
                t + 1, tf.matmul(self.css.read(t), w_cs)
            )
            self.input_hos_hs = self.input_hos_hs.write(
                t + 1, tf.matmul(self.hos.read(t), self.w_hos_hs)
            )

            self.input_sem = self.input_sem.write(
                t + 1,
                self.tau
                * (
                    self.input_hps_hs.read(t + 1)
                    # + self.input_sem_ss.read(t + 1)
                    + self.input_css_cs.read(t + 1)
                    + self.input_hos_hs.read(t + 1)
                    + bias_s
                )
                + (1 - self.tau) * self.input_sem.read(t),
            )

            ##### Phonology layer #####
            self.input_hsp_hp = self.input_hsp_hp.write(
                t + 1, tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            )
            self.input_pho_pp = self.input_pho_pp.write(
                t + 1, tf.matmul(self.pho.read(t), w_pp)
            )
            self.input_cpp_cp = self.input_cpp_cp.write(
                t + 1, tf.matmul(self.cpp.read(t), w_cp)
            )
            self.input_hop_hp = self.input_hop_hp.write(
                t + 1, tf.matmul(self.hop.read(t), self.w_hop_hp)
            )

            self.input_pho = self.input_pho.write(
                t + 1,
                self.tau
                * (
                    self.input_cpp_cp.read(t + 1)
                    # self.input_hsp_hp.read(t + 1) [LESION]
                    # + self.input_pho_pp.read(t + 1)
                    + self.input_hop_hp.read(t + 1)
                    + bias_p
                )
                + (1 - self.tau) * self.input_pho.read(t),
            )

            ##### Hidden layer (PS) #####
            self.input_hps = self.input_hps.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps)
                + (1 - self.tau) * self.input_hps.read(t),
            )

            ##### Hidden layer (SP) #####
            self.input_hsp = self.input_hsp.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp)
                + (1 - self.tau) * self.input_hsp.read(t),
            )

            ##### Semantic Cleanup layer #####
            self.input_css = self.input_css.write(
                t + 1,
                self.tau * (tf.matmul(self.sem.read(t), w_sc) + bias_css)
                + (1 - self.tau) * self.input_css.read(t),
            )

            ##### Phonology Cleanup layer #####
            self.input_cpp = self.input_cpp.write(
                t + 1,
                self.tau * (tf.matmul(self.pho.read(t), w_pc) + bias_cpp)
                + (1 - self.tau) * self.input_cpp.read(t),
            )

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def _inject_noise(self, x: tf.Tensor, noise_sd: float) -> tf.Tensor:
        """Inject Gaussian noise if noise_sd > 0"""
        if noise_sd > 0:
            noise = K.random_normal(shape=K.shape(x), mean=0.0, stddev=noise_sd)
            return x + noise
        else:
            return x

    def _init_input_array(
        self, name: str, shape: tuple, value: float = 0
    ):  # Init at zero

        setattr(
            self,
            name,
            getattr(self, name).write(
                0, tf.constant(value, dtype=tf.float32, shape=shape)
            ),
        )

    def _init_all_tensor_arrays(self):
        """At the beginning of all tasks, reset all time related tensor arrays"""

        # Recreate tensor array for safety
        for x in self.ALL_ARRAY_NAMES:
            setattr(
                self,
                x,
                tf.TensorArray(
                    tf.float32, size=self.n_timesteps + 1, clear_after_read=False
                ),
            )

        # Set inputs
        [
            self._init_input_array(x, shape=self.shapes[x])
            for x in self.INPUT_ARRAY_NAMES
        ]

        # Set activations to init value
        [self._update_activations(0, x) for x in self.ACTIVATION_ARRAY_NAMES]

    def _inject_noise_to_all_pho(self, training):
        """inject noise to all PHO relate weights and biases"""
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

        return w_pp, w_pc, w_cp, bias_cpp, bias_p

    def _inject_noise_to_all_sem(self, training):
        """inject noise to all SEM relate weights and biases"""
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

        return w_ss, w_sc, w_cs, bias_css, bias_s

    def _update_activations(self, timetick: int, act: str):
        """Update the activation"""
        sum_input = getattr(self, f"input_{act}").read(timetick)
        activation = self.activation(sum_input)

        setattr(self, act, getattr(self, act).write(timetick, activation))

    def _package_output(self, training):

        output_dict = K.in_train_phase(
            {
                k: getattr(self, k).stack()[-self.inject_error_ticks :]
                for k in self.ALL_ARRAY_NAMES
            },
            {
                k: getattr(self, k).stack()[-self.output_ticks :]
                for k in self.ALL_ARRAY_NAMES
            },
            training=training,
        )

        # Close all array to release memeory
        [getattr(self, x).close() for x in self.ALL_ARRAY_NAMES]
        return output_dict

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
