import tensorflow as tf
import tensorflow.keras.backend as K

# Create dictionary for trainable weight & biases related to each task (only affect weight update step)
## Important: Due to model complexity, it seems that "trainable" flag cannot be use to turn on/off training using Keras training API
## Therefore, it must use custom training loop to control which matrix to perform gradient descent
## Since there are 4 sets of hidden layers and 2 sets of cleanup units,
## when refering to hidden, we need to state the exact layer in this format: h{from}{to} in weights
## when refering to cleanup, we need to use this format in biases: bias_c{from}{to}


WEIGHTS_AND_BIASES = {}
WEIGHTS_AND_BIASES["pho_sem"] = (
    "w_hps_ph",
    "w_hps_hs",
    "w_sc",
    "w_cs",
    "bias_hps",
    "bias_s",
    "bias_css",
)
WEIGHTS_AND_BIASES["sem_pho"] = (
    "w_hsp_sh",
    "w_hsp_hp",
    "w_pc",
    "w_cp",
    "bias_hsp",
    "bias_p",
    "bias_cpp",
)
WEIGHTS_AND_BIASES["pho_pho"] = ("w_pc", "w_cp", "bias_p", "bias_cpp")
WEIGHTS_AND_BIASES["sem_sem"] = ("w_sc", "w_cs", "bias_s", "bias_css")
WEIGHTS_AND_BIASES["ort_sem"] = (
    "w_dos",
    "w_hos_oh",
    "w_hos_hs",
    "bias_hos",
    "bias_s",
    "w_sc",
    "w_cs",
    "bias_css",
)
WEIGHTS_AND_BIASES["ort_pho"] = (
    "w_dop",
    "w_hop_oh",
    "w_hop_hp",
    "bias_hop",
    "bias_p",
    "w_pc",
    "w_cp",
    "bias_cpp",
)


# WEIGHTS_AND_BIASES["ort_sem"] = ("w_hos_oh", "w_hos_hs", "bias_hos", "bias_s")
# WEIGHTS_AND_BIASES["ort_pho"] = ("w_hop_oh", "w_hop_hp", "bias_hop", "bias_p")
# WEIGHTS_AND_BIASES["ort_sem"] = ("w_hos_oh", "w_hos_hs", "bias_hos")
# WEIGHTS_AND_BIASES["ort_pho"] = ("w_hop_oh", "w_hop_hp", "bias_hop")

WEIGHTS_AND_BIASES["triangle"] = (
    "w_dop",
    "w_dos",
    "w_hos_oh",
    "w_hos_hs",
    "bias_hos",
    "w_hop_oh",
    "w_hop_hp",
    "bias_hop",
    # "bias_p",
    # "bias_s",
    # "w_pc",
    # "w_cp",
    # "bias_cpp",
    # "w_sc",
    # "w_cs",
    # "bias_css",
    # "w_hps_ph",
    # "w_hps_hs",
    # "bias_hps",
    # "w_hsp_sh",
    # "w_hsp_hp",
    # "bias_hsp",
)

WEIGHTS_AND_BIASES["exp_os_ff"] = ("w_hos_oh", "w_hos_hs", "bias_hos", "bias_s")

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


IN_OUT["exp_os_ff"] = ("ort", "sem")


# LAYERS contains all layers' name in the model
LAYERS = ("pho", "sem", "hos", "hop", "css", "cpp", "hps", "hsp")

# CONNECTIONS describe what is the immediate connection of a given layer (Not used in model, only for reference.)
# TODO: Modularize the layers and use this dictionary to build the model.
CONNECTIONS = {}
CONNECTIONS["hos"] = ("w_hos_oh", "bias_hos")
CONNECTIONS["hop"] = ("w_hop_oh", "bias_hop")
CONNECTIONS["sem"] = ("w_dos", "w_hos_hs", "w_hps_hs", "w_cs", "bias_s")
CONNECTIONS["pho"] = ("w_dop", "w_hop_hp", "w_hsp_hp", "w_cp", "bias_p")
CONNECTIONS["css"] = ("w_sc", "bias_css")
CONNECTIONS["cpp"] = ("w_pc", "bias_cpp")
CONNECTIONS["hps"] = ("w_hps_ph", "bias_hps")
CONNECTIONS["hsp"] = ("w_hsp_sh", "bias_hsp")


class TriangleModel(tf.keras.Model):
    """Model object with full output in dictionary format.

    To predict (doing a forward pass), do not use model.predict(x), Use model(x) to instead.
    """

    INPUT_ARRAY_NAMES = (
        "input_hos",  # sum time-averaged input + bias
        "input_hop",
        "input_hps",
        "input_hsp",
        "input_css",
        "input_cpp",
        "input_sem",
        "input_pho",
        "input_hps_hs",  # raw time-averaged inputs
        "input_css_cs",
        "input_hos_hs",
        "input_hsp_hp",
        "input_cpp_cp",
        "input_hop_hp",
        "input_dop_op",
        "input_dos_os",
    )

    ACTIVATION_ARRAY_NAMES = ("hos", "hop", "hps", "hsp", "css", "cpp", "sem", "pho")

    ALL_ARRAY_NAMES = INPUT_ARRAY_NAMES + ACTIVATION_ARRAY_NAMES

    def __init__(self, cfg, name="triangle", **kwargs):
        """Initialize the model.
        IMPORTANT: Do not set active task while initializing the model. It will trigger a inf. recursion.
        """
        super().__init__(**kwargs)
        self.cfg = cfg  # Store the entire config object

        # Unpack config to model level variables
        for key, value in cfg.__dict__.items():
            setattr(self, key, value)  # copy all config values to model

        # Inferred variable (@property) needed to unpack manually
        self.n_timesteps = cfg.n_timesteps
        self.output_ticks = cfg.output_ticks
        self.activation = tf.keras.activations.get(self.activation)

        # For clarity, explicitly create mapping between "task name (for user)" and "class method"
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
            "exp_os_ff": self.task_ort_sem_ff,
        }

    @property
    def UNIT_SIZE(self) -> dict:
        """Get the unit size for each layer for initializing input array."""
        return {
            "input_hos": self.hidden_os_units,
            "input_hop": self.hidden_op_units,
            "input_sem": self.sem_units,
            "input_pho": self.pho_units,
            "input_hps": self.hidden_ps_units,
            "input_hsp": self.hidden_sp_units,
            "input_css": self.sem_cleanup_units,
            "input_cpp": self.pho_cleanup_units,
            "input_hps_hs": self.sem_units,
            "input_css_cs": self.sem_units,
            "input_hos_hs": self.sem_units,
            "input_hsp_hp": self.pho_units,
            "input_cpp_cp": self.pho_units,
            "input_hop_hp": self.pho_units,
            "input_dop_op": self.pho_units,
            "input_dos_os": self.sem_units,
        }

    def build(self, input_shape=None):
        """Build entire model's weights and biases."""

        weight_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        bias_initializer = tf.constant_initializer(value=0.0)

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

        self.w_dos = self.add_weight(
            shape=(self.ort_units, self.sem_units),
            name="w_dos",
            initializer=weight_initializer,
            trainable=True,
        )

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
        self.w_dop = self.add_weight(
            shape=(self.ort_units, self.pho_units),
            name="w_dop",
            initializer=weight_initializer,
            trainable=True,
        )

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

    def set_active_task(self, task: str):
        """User interface for switching task."""
        self.active_task = task

    def call(self, inputs, training=None) -> dict:
        """Call active task when running model().

        Arguments:
            inputs: model input
            input dimension: [timestep (input should be identical across timestep), item_in_batch, input_unit]

        return: a dictionary of input and activation depending on task
        """

        return self.tasks[self.active_task](inputs, training)

    def task_pho_sem(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (PS) #####
            new_inputs["input_hps"] = (
                tf.matmul(inputs[t], self.w_hps_ph) + self.bias_hps
            )

            ##### Semantic layer #####
            new_inputs["input_hps_hs"] = tf.matmul(self.hps.read(t), self.w_hps_hs)
            new_inputs["input_css_cs"] = tf.matmul(self.css.read(t), w_cs)

            new_inputs["input_sem"] = (
                new_inputs["input_hps_hs"] + new_inputs["input_css_cs"] + bias_s
            )

            ##### Semantic Cleanup layer #####
            new_inputs["input_css"] = tf.matmul(self.sem.read(t), w_sc) + bias_css

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            [self._update_activations(t + 1, x) for x in ("hps", "sem", "css")]

        return self._package_output(training=training)

    def task_sem_sem(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ### Semantic layer###
            new_inputs["input_css_cs"] = tf.matmul(self.css.read(t), w_cs)
            new_inputs["input_sem"] = new_inputs["input_css_cs"] + bias_s

            ##### Semantic Cleanup layer #####
            new_inputs["input_css"] = tf.matmul(self.sem.read(t), w_sc) + bias_css

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            if t < 8:
                # Clamp activation for the first 8 timesteps
                self.sem = self.sem.write(t + 1, inputs[t])
            else:
                self.sem = self.sem.write(
                    t + 1, self.activation(self.input_sem.read(t + 1))
                )

            self._update_activations(t + 1, "css")

        return self._package_output(training=training)

    def task_sem_pho(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)

            ##### Hidden layer (SP) #####
            new_inputs["input_hsp"] = (
                tf.matmul(inputs[t], self.w_hsp_sh) + self.bias_hsp
            )

            ##### Phonology layer #####
            new_inputs["input_hsp_hp"] = tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            new_inputs["input_cpp_cp"] = tf.matmul(self.cpp.read(t), w_cp)

            new_inputs["input_pho"] = (
                new_inputs["input_hsp_hp"] + new_inputs["input_cpp_cp"] + bias_p
            )

            ##### Phonology Cleanup layer #####
            new_inputs["input_cpp"] = tf.matmul(self.pho.read(t), w_pc) + bias_cpp

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            [self._update_activations(t + 1, x) for x in ("hsp", "pho", "cpp")]

        return self._package_output(training=training)

    def task_pho_pho(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)

            # Phonological unit
            new_inputs["input_cpp_cp"] = tf.matmul(self.cpp.read(t), w_cp)
            new_inputs["input_pho"] = new_inputs["input_cpp_cp"] + bias_p

            ##### Phonology Cleanup layer #####
            new_inputs["input_cpp"] = tf.matmul(self.pho.read(t), w_pc) + bias_cpp

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
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
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            new_inputs["input_hos"] = (
                tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos
            )

            ##### Semantic layer #####
            new_inputs["input_css_cs"] = tf.matmul(self.css.read(t), w_cs)
            new_inputs["input_hos_hs"] = tf.matmul(self.hos.read(t), self.w_hos_hs)
            new_inputs["input_dos_os"] = tf.matmul(input[t], self.w_dos)  # Direct OS
            new_inputs["input_sem"] = (
                new_inputs["input_css_cs"]
                + new_inputs["input_hos_hs"]
                + new_inputs["input_dos_os"]
                + bias_s
            )

            ##### Semantic Cleanup layer #####
            new_inputs["input_css"] = tf.matmul(self.sem.read(t), w_sc) + bias_css

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activation
            [self._update_activations(t + 1, x) for x in ("sem", "css", "hos")]

        return self._package_output(training=training)

    def task_ort_sem_ff(self, inputs, training=None):
        """OS feedforward task"""

        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)
            ##### Hidden layer (OS) #####
            new_inputs["input_hos"] = (
                tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos
            )

            ##### Semantic layer #####
            new_inputs["input_hos_hs"] = tf.matmul(self.hos.read(t), self.w_hos_hs)
            new_inputs["input_sem"] = new_inputs["input_hos_hs"] + bias_s

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activation
            [self._update_activations(t + 1, x) for x in ("sem", "hos")]

        return self._package_output(training=training)

    def task_ort_pho(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)

            ##### Hidden layer (OP) #####
            new_inputs["input_hop"] = (
                tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop
            )

            ##### Phonology layer #####
            new_inputs["input_cpp_cp"] = tf.matmul(self.cpp.read(t), w_cp)
            new_inputs["input_hop_hp"] = tf.matmul(self.hop.read(t), self.w_hop_hp)
            new_inputs["input_dop_op"] = tf.matmul(inputs[t], self.w_dop)

            new_inputs["input_pho"] = (
                new_inputs["input_cpp_cp"]
                + new_inputs["input_hop_hp"]
                + new_inputs["input_dop_op"]
                + bias_p
            )

            ##### Phonology Cleanup layer #####
            new_inputs["input_cpp"] = tf.matmul(self.pho.read(t), w_pc) + bias_cpp

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            [self._update_activations(t + 1, x) for x in ("pho", "cpp", "hop")]

        return self._package_output(training=training)

    def task_triangle(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            new_inputs["input_hos"] = (
                tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos
            )

            ##### Hidden layer (OP) #####
            new_inputs["input_hop"] = (
                tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop
            )

            ##### Semantic layer #####
            new_inputs["input_hps_hs"] = tf.matmul(self.hps.read(t), self.w_hps_hs)
            new_inputs["input_css_cs"] = tf.matmul(self.css.read(t), w_cs)
            new_inputs["input_hos_hs"] = tf.matmul(self.hos.read(t), self.w_hos_hs)
            new_inputs["input_dos_os"] = tf.matmul(inputs[t], self.w_dos)

            new_inputs["input_sem"] = (
                new_inputs["input_hps_hs"]
                + new_inputs["input_css_cs"]
                + new_inputs["input_hos_hs"]
                + new_inputs["input_dos_os"]
                + bias_s
            )

            ##### Phonology layer #####
            new_inputs["input_hsp_hp"] = tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            new_inputs["input_cpp_cp"] = tf.matmul(self.cpp.read(t), w_cp)
            new_inputs["input_hop_hp"] = tf.matmul(self.hop.read(t), self.w_hop_hp)
            new_inputs["input_dop_op"] = tf.matmul(inputs[t], self.w_dop)

            new_inputs["input_pho"] = (
                new_inputs["input_hsp_hp"]
                + new_inputs["input_cpp_cp"]
                + new_inputs["input_hop_hp"]
                + new_inputs["input_dop_op"]
                + bias_p
            )

            ##### Hidden layer (PS) #####
            new_inputs["input_hps"] = (
                tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps
            )

            ##### Hidden layer (SP) #####
            new_inputs["input_hsp"] = (
                tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp
            )

            ##### Semantic Cleanup layer #####
            new_inputs["input_css"] = tf.matmul(self.sem.read(t), w_sc) + bias_css

            ##### Phonology Cleanup layer #####
            new_inputs["input_cpp"] = tf.matmul(self.pho.read(t), w_pc) + bias_cpp

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def experimental_task_ops(self, inputs, training=None):
        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            new_inputs["input_hos"] = (
                tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos
            )

            ##### Hidden layer (OP) #####
            new_inputs["input_hop"] = (
                tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop
            )

            ##### Semantic layer #####
            new_inputs["input_hps_hs"] = tf.matmul(self.hps.read(t), self.w_hps_hs)
            new_inputs["input_css_cs"] = tf.matmul(self.css.read(t), w_cs)
            # new_inputs["input_hos_hs"] = tf.matmul(self.hos.read(t), self.w_hos_hs)

            new_inputs["input_sem"] = (
                new_inputs["input_hps_hs"]
                + new_inputs["input_css_cs"]
                # + new_inputs['input_hos_hs'] [LESION]
                + bias_s
            )

            ##### Phonology layer #####
            new_inputs["input_hsp_hp"] = tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            new_inputs["input_cpp_cp"] = tf.matmul(self.cpp.read(t), w_cp)
            new_inputs["input_hop_hp"] = tf.matmul(self.hop.read(t), self.w_hop_hp)
            new_inputs["input_dop_op"] = tf.matmul(inputs[t], self.w_dop)

            new_inputs["input_pho"] = (
                new_inputs["input_hsp_hp"]
                + new_inputs["input_cpp_cp"]
                + new_inputs["input_hop_hp"]
                + new_inputs["input_dop_op"]
                + bias_p
            )

            ##### Hidden layer (PS) #####
            new_inputs["input_hps"] = (
                tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps
            )

            ##### Hidden layer (SP) #####
            new_inputs["input_hsp"] = (
                tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp
            )

            ##### Semantic Cleanup layer #####
            new_inputs["input_css"] = tf.matmul(self.sem.read(t), w_sc) + bias_css

            ##### Phonology Cleanup layer #####
            new_inputs["input_cpp"] = tf.matmul(self.pho.read(t), w_pc) + bias_cpp

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def experimental_task_osp(self, inputs, training=None):
        """Lesion triangle model with HOP damaged"""

        batch_size = inputs[0].shape[0]
        self._init_all_tensor_arrays(batch_size)

        # Recurrent structure over time ticks (Time averaged input)
        new_inputs = {}
        for t in range(self.n_timesteps):
            # Inject fresh white noise in each tick to weights and biases
            # If noise is 0 or at evaluation phase (track by training flag), it will do nothing.
            w_pc, w_cp, bias_cpp, bias_p = self._inject_noise_to_all_pho(training)
            w_sc, w_cs, bias_css, bias_s = self._inject_noise_to_all_sem(training)

            ##### Hidden layer (OS) #####
            new_inputs["input_hos"] = (
                tf.matmul(inputs[t], self.w_hos_oh) + self.bias_hos
            )

            ##### Hidden layer (OP) #####
            new_inputs["input_hop"] = (
                tf.matmul(inputs[t], self.w_hop_oh) + self.bias_hop
            )

            ##### Semantic layer #####
            new_inputs["input_hps_hs"] = tf.matmul(self.hps.read(t), self.w_hps_hs)
            new_inputs["input_css_cs"] = tf.matmul(self.css.read(t), w_cs)
            new_inputs["input_hos_hs"] = tf.matmul(self.hos.read(t), self.w_hos_hs)
            new_inputs["input_dos_os"] = tf.matmul(inputs[t], self.w_dos)

            new_inputs["input_sem"] = (
                new_inputs["input_hps_hs"]
                + new_inputs["input_css_cs"]
                + new_inputs["input_hos_hs"]
                + new_inputs["input_dos_os"]
                + bias_s
            )

            ##### Phonology layer #####
            new_inputs["input_hsp_hp"] = tf.matmul(self.hsp.read(t), self.w_hsp_hp)
            new_inputs["input_cpp_cp"] = tf.matmul(self.cpp.read(t), w_cp)
            # new_inputs["input_hop_hp"] = tf.matmul(self.hop.read(t), self.w_hop_hp)

            new_inputs["input_pho"] = (
                new_inputs["input_hsp_hp"]
                + new_inputs["input_cpp_cp"]
                # + new_inputs['input_hop_hp'] [LESION]
                + bias_p
            )

            ##### Hidden layer (PS) #####
            new_inputs["input_hps"] = (
                tf.matmul(self.pho.read(t), self.w_hps_ph) + self.bias_hps
            )

            ##### Hidden layer (SP) #####
            new_inputs["input_hsp"] = (
                tf.matmul(self.sem.read(t), self.w_hsp_sh) + self.bias_hsp
            )

            ##### Semantic Cleanup layer #####
            new_inputs["input_css"] = tf.matmul(self.sem.read(t), w_sc) + bias_css

            ##### Phonology Cleanup layer #####
            new_inputs["input_cpp"] = tf.matmul(self.pho.read(t), w_pc) + bias_cpp

            # Process time averaged inputs
            [self._tai(name, new_input, t) for name, new_input in new_inputs.items()]

            # Update activations
            [self._update_activations(t + 1, x) for x in self.ACTIVATION_ARRAY_NAMES]

        return self._package_output(training=training)

    def _tai(self, input_array_name: str, new_input: tf.Variable, t: int) -> None:
        """Perform on step of time averaged input.
        Formula: tai_input = tau * new_input + (1 - tau) * last_input
        where new_input is at t+1 and last_input is at t.
        """
        input_array = getattr(self, input_array_name)
        last_input = input_array.read(t)
        tai_input = (1 - self.tau) * last_input + self.tau * new_input
        setattr(self, input_array_name, input_array.write(t + 1, tai_input))

    def _inject_noise(self, x: tf.Tensor, noise_sd: float) -> tf.Tensor:
        """Inject Gaussian noise if noise_sd > 0"""
        if noise_sd > 0:
            noise = K.random_normal(shape=K.shape(x), mean=0.0, stddev=noise_sd)
            return x + noise
        else:
            return x

    def _init_input_array(self, name: str, batch_size: int, init_value: float = 0.0):
        unit_size = self.UNIT_SIZE[name]
        init_value = tf.constant(
            init_value, dtype=tf.float32, shape=(batch_size, unit_size)
        )
        setattr(self, name, getattr(self, name).write(index=0, value=init_value))

    def _init_all_tensor_arrays(self, batch_size: int):
        """At the beginning of all tasks, reset all time related tensor arrays"""

        # Recreate tensor array for safety
        for x in self.ALL_ARRAY_NAMES:
            arr = tf.TensorArray(
                tf.float32, size=self.n_timesteps + 1, clear_after_read=False
            )
            setattr(self, x, arr)

        # Set inputs
        [self._init_input_array(x, batch_size) for x in self.INPUT_ARRAY_NAMES]

        # Set activations to init value
        [self._update_activations(0, x) for x in self.ACTIVATION_ARRAY_NAMES]

    def _inject_noise_to_all_pho(self, training):
        """inject noise to all PHO relate weights and biases"""

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

        return w_pc, w_cp, bias_cpp, bias_p

    def _inject_noise_to_all_sem(self, training):
        """inject noise to all SEM relate weights and biases"""

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

        return w_sc, w_cs, bias_css, bias_s

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
