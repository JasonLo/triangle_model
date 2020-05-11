import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers, Model
from tensorflow.keras.layers import Layer, Input
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output


### Zero-error-radius related:

from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import nn, clip_ops, math_ops, array_ops
from tensorflow.keras.backend import epsilon

def _constant_to_tensor(x, dtype):
    return constant_op.constant(x, dtype=dtype)

def _backtrack_identity(tensor):
    while tensor.op.type == "Identity":
        tensor = tensor.op.inputs[0]
    return tensor

def zer_replace(target, output, zero_error_raidus):
    """Replace output by target if value within zero-error-radius
    """
    zeros = tf.zeros_like(output, dtype=output.dtype)
    zer_threshold = tf.constant(zero_error_raidus)
    zer_mask = tf.math.less(tf.math.abs(output - target), zer_threshold)
    zer_output = tf.where(zer_mask, target, output)
    return zer_output

def zer_bce(target, output):
    if not isinstance(output, (ops.EagerTensor, variables_module.Variable)):
        output = _backtrack_identity(output)

    # Clip with a tiny constant to avoid zero division
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Replace output by target if value within zero error radius of 0.1
    zer_output = zer_replace(target, output, 0.1)

    # Compute cross entropy from probabilities.
    bce = target * math_ops.log(zer_output + epsilon())
    bce += (1 - target) * math_ops.log(1 - zer_output + epsilon())
    return -bce

###

# Older Semantic
# def input_s(e, t, f, i,
#             gf=4, gi=1,
#             kf=100, ki=100,
#             hf=5, hi=5,
#             tmax=3.8,
#             mf=4.4743, sf=2.4578,
#             mi=4.1988, si=1.0078):

#     zf = (np.log(f + 2) - mf) / sf
#     numer_f = gf * e * (zf + hf)
#     denom_f = e * (zf + hf) + kf

#     zi = (i - mi) / si
#     numer_i = gi * e * (zi + hi)
#     denom_i = e * (zi + hi) + ki

#     return (t / tmax) * ((numer_f / denom_f) + (numer_i / denom_i))

def input_s(e, t, f, i, gf, gi, kf, ki, tmax=3.8,
            mf=4.4743, sf=2.4578, mi=4.1988, si=1.0078, hf=0, hi=0):
    # Semantic refresh V1

    numer_f = gf * e * np.log(f+2)
    denom_f = e * np.log(f+2) + kf

    return (t/tmax)*(numer_f/denom_f)

# def input_s(e,
#             t,
#             f,
#             i,
#             gf,
#             gi,
#             kf,
#             ki,
#             tmax=3.8,
#             mf=4.4743,
#             sf=2.4578,
#             mi=4.1988,
#             si=1.0078,
#             hf=5,
#             hi=5):
#     # Semantic refresh V2
#     numer_f = gf * np.sqrt(e) * np.log(f + 2)
#     denom_f = np.sqrt(e) * np.log(f + 2) + kf

#     return (t / tmax) * (numer_f / denom_f)


# def input_s(
#     e,
#     t,
#     f,
#     i,
#     gf,
#     gi,
#     kf,
#     ki,
#     tmax=3.8,
#     mf=4.4743,
#     sf=2.4578,
#     mi=4.1988,
#     si=1.0078,
#     hf=5,
#     hi=5
# ):
#     # Semantic refresh V3
#     numer_e = gf * e
#     denom_e = e + kf

#     return (t / tmax) * ((numer_e / denom_e) + 0.1 * np.log(f + 2))


class rnn(Layer):
    """
    Main time-averaged input implementation based on Plaut 96 (Fig 12.)
    With additional attractor (cleanup) network
    Option to use semantic input by cfg.use_semantic == True
    Semantic equation can be change in modeling.input_s()
    """
    # Use keras rnn layer seems more efficient, maybe upgrade later...
    def __init__(self, cfg, input_p_dignostic=False, name='rnn', **kwargs):

        super(rnn, self).__init__(**kwargs)

        self.cfg = cfg
        self.input_p_dignostic = input_p_dignostic

        self.rnn_activation = activations.get(self.cfg.rnn_activation)
        
        
        if self.cfg.regularizer_const == None:
            self.weight_regularizer = None
        else:
            self.weight_regularizer = regularizers.l2(cfg.regularizer_const)
            
        self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=self.cfg.w_initializer, seed=self.cfg.rng_seed)
        
        self.w_oh = self.add_weight(
            name='w_oh',
            shape=(self.cfg.input_dim, self.cfg.hidden_units),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_hp = self.add_weight(
            name='w_hp',
            shape=(self.cfg.hidden_units, self.cfg.output_dim),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_pp = self.add_weight(
            name='w_pp',
            shape=(self.cfg.output_dim, self.cfg.output_dim),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_pc = self.add_weight(
            name='w_pc',
            shape=(self.cfg.output_dim, self.cfg.cleanup_units),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_cp = self.add_weight(
            name='w_cp',
            shape=(self.cfg.cleanup_units, self.cfg.output_dim),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.bias_h = self.add_weight(
            shape=(self.cfg.hidden_units, ),
            name='bias_h',
            initializer='zeros',
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.bias_p = self.add_weight(
            shape=(self.cfg.output_dim, ),
            name='bias_p',
            initializer='zeros',
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.bias_c = self.add_weight(
            shape=(self.cfg.cleanup_units, ),
            name='bias_c',
            initializer='zeros',
            regularizer=self.weight_regularizer,
            trainable=True
        )

    def call(self, inputs):
        """
        If input_p_dignostic = True, it will output input_p instead of act_p (for troubleshooting)
        Hack for complying keras.layers.concatenate() format
        Dimension note: (batch, timestep, input_dim)
        Spliting input_dim below (index = 2)
        """

        if self.cfg.use_semantic == True:
            o_input, s_input = tf.split(
                inputs, [self.cfg.input_dim, self.cfg.output_dim], 2
            )
        else:
            o_input = inputs

        ### Trial level init ###
        self.input_h_list = []
        self.input_p_list = []
        self.input_c_list = []

        self.act_h_list = []
        self.act_p_list = []
        self.act_c_list = []

        # Set inputs to 0
        self.input_h_list.append(
            tf.zeros((1, self.cfg.hidden_units), dtype=tf.float32)
        )
        self.input_p_list.append(
            tf.zeros((1, self.cfg.output_dim), dtype=tf.float32)
        )
        self.input_c_list.append(
            tf.zeros((1, self.cfg.cleanup_units), dtype=tf.float32)
        )

        # Set activations to 0.5
        self.act_h_list.append(self.input_h_list[0] + 0.5)
        self.act_p_list.append(self.input_p_list[0] + 0.5)
        self.act_c_list.append(self.input_c_list[0] + 0.5)

        for t in range(1, self.cfg.n_timesteps + 1):
            # Inject noise to weights in each time step
            # Method 1: Inject noise at each time step with reset
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

            if self.cfg.w_pc_noise != 0:
                w_pc = self.inject_noise(self.w_pc, self.cfg.w_pc_noise)
            else:
                w_pc = self.w_pc

            if self.cfg.w_cp_noise != 0:
                w_cp = self.inject_noise(self.w_cp, self.cfg.w_cp_noise)
            else:
                w_cp = self.w_cp

            ##### Hidden layer #####
            oh = tf.matmul(o_input[:, t - 1, :], w_oh)
            mem_h = self.input_h_list[t - 1]
            h = self.cfg.tau * (oh + self.bias_h) + (1 - self.cfg.tau) * mem_h

            self.input_h_list.append(h)
            self.act_h_list.append(self.rnn_activation(h))

            ##### Phonology layer #####
            hp = tf.matmul(self.act_h_list[t - 1], w_hp)
            pp = tf.matmul(self.act_p_list[t - 1], w_pp)
            
#             # Zero diagonal lock
#             pp = tf.matmul(
#                 self.act_p_list[t - 1],
#                 tf.linalg.set_diag(w_pp, tf.zeros(self.cfg.output_dim))
#             )  
            cp = tf.matmul(self.act_c_list[t - 1], w_cp)

            mem_p = self.input_p_list[t - 1]

            if self.cfg.use_semantic == True:  # Inject semantic input
                sp = s_input[:, t - 1, :]
            else:
                sp = 0

            p = self.cfg.tau * (hp + pp + cp + sp +
                                self.bias_p) + (1 - self.cfg.tau) * mem_p

            self.input_p_list.append(p)
            self.act_p_list.append(self.rnn_activation(p))

            ##### Cleanup layer #####
            pc = tf.matmul(self.act_p_list[t - 1], w_pc)
            mem_c = self.input_c_list[t - 1]
            c = self.cfg.tau * (pc + self.bias_c) + (1 - self.cfg.tau) * mem_c

            self.input_c_list.append(c)
            self.act_c_list.append(self.rnn_activation(c))

        if self.input_p_dignostic == True:
            return self.input_p_list[-self.cfg.output_ticks:]
        else:
            return self.act_p_list[-self.cfg.output_ticks:]

    def inject_noise(self, x, noise_sd):
        noise = K.random_normal(shape=K.shape(x), mean=0., stddev=noise_sd)
        return x + noise

    def compute_output_shape(self):
        return tensor_shape.as_shape(
            [1, self.cfg.output_dim] + self.cfg.output_ticks
        )

    def get_config(self):
        config = {'custom_cfg': self.cfg}
        base_config = super(rnn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class attractor_rnn(Layer):
    """
    Recurrent Attractor Layer
    
    From HS04:
    The phonological form of the target word was clamped on the phonological units for 2.66 units of time. 
    Then a target signal was provided for the next 1.33 units of time, in which the network was required 
    to retain the phonological pattern in the absence of external clamping.
    
    We have tau = 0.2 instead of 0.33 --> 2.66 units of time ~= 14 steps (2.8 units)
    In the last 6 steps, we provide training signal for backprob
    
    Noise can also be added in each weight matrix by using cfg.w_xx_noise
    Which will add a Gaussian noise (SD = noise level) at each time step
    """
    def __init__(self, cfg, clamp_steps=14, **kwargs):
        super(attractor_rnn, self).__init__(**kwargs)
        self._name = 'rnn'
        self.cfg = cfg
        self.clamp_steps = clamp_steps
        self.rnn_activation = activations.get(self.cfg.rnn_activation)
        self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=self.cfg.w_initializer, seed=self.cfg.rng_seed)

    def build(self, input_shape, **kwargs):

        self.w_pp = self.add_weight(
            name='w_pp',
            shape=(self.cfg.output_dim, self.cfg.output_dim),
            initializer=self.w_initializer,
            trainable=True
        )

        self.w_pc = self.add_weight(
            name='w_pc',
            shape=(self.cfg.output_dim, self.cfg.cleanup_units),
            initializer=self.w_initializer,
            trainable=True
        )

        self.w_cp = self.add_weight(
            name='w_cp',
            shape=(self.cfg.cleanup_units, self.cfg.output_dim),
            initializer=self.w_initializer,
            trainable=True
        )

        self.bias_p = self.add_weight(
            shape=(self.cfg.output_dim, ),
            name='bias_p',
            initializer='zeros',
            trainable=True
        )

        self.bias_c = self.add_weight(
            shape=(self.cfg.cleanup_units, ),
            name='bias_c',
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):

        import tensorflow as tf

        ### Trial level init ###
        self.input_p_list = []
        self.input_c_list = []

        self.act_p_list = []
        self.act_c_list = []

        # Initialize input at step 0 to 3*input
        self.input_p_list.append(inputs * 3)
        self.input_c_list.append(
            tf.zeros((1, self.cfg.cleanup_units), dtype=tf.float32)
        )

        # Initialize activations
        self.act_p_list.append(self.rnn_activation(self.input_p_list[0]))
        self.act_c_list.append(self.input_c_list[0] + 0.5)

        for t in range(1, self.cfg.n_timesteps + 1):
            
            if self.cfg.w_pp_noise != 0:
                w_pp = self.inject_noise(self.w_pp, self.cfg.w_pp_noise)
            else:
                w_pp = self.w_pp

            if self.cfg.w_pc_noise != 0:
                w_pc = self.inject_noise(self.w_pc, self.cfg.w_pc_noise)
            else:
                w_pc = self.w_pc

            if self.cfg.w_cp_noise != 0:
                w_cp = self.inject_noise(self.w_cp, self.cfg.w_cp_noise)
            else:
                w_cp = self.w_cp
            

            # ##### Phonology layer #####
            pp = tf.matmul(self.act_p_list[t - 1], w_pp)
            
#             # Zero diagonal lock     
#             pp = tf.matmul(
#                 self.act_p_list[t - 1],
#                 tf.linalg.set_diag(w_pp, tf.zeros(self.cfg.output_dim))
#             )  
            cp = tf.matmul(self.act_c_list[t - 1], w_cp)

            mem_p = self.input_p_list[t - 1]
            p = self.cfg.tau * (pp + cp +
                                self.bias_p) + (1 - self.cfg.tau) * mem_p

            self.input_p_list.append(p)

            if self.cfg.n_timesteps <= self.clamp_steps:
                act_p = inputs
            else:
                act_p = self.rnn_activation(p)

            self.act_p_list.append(act_p)

            ##### Cleanup layer #####
            pc = tf.matmul(self.act_p_list[t - 1], w_pc)

            mem_c = self.input_c_list[t - 1]
            c = self.cfg.tau * (pc + self.bias_c) + (1 - self.cfg.tau) * mem_c

            self.input_c_list.append(c)
            self.act_c_list.append(self.rnn_activation(c))

        return self.act_p_list[self.clamp_steps + 1:
                              ]  # Can get forgetting curve?
    
    def inject_noise(self, x, noise_sd):
        noise = K.random_normal(shape=K.shape(x), mean=0., stddev=noise_sd)
        return x + noise
    
    def compute_output_shape(self):
        n = self.cfg.n_timesteps - self.clamp_steps
        return tensor_shape.as_shape([1, self.cfg.output_dim] + n)

    def get_config(self):
        config = {'custom_cfg': self.cfg}
        base_config = super(attractor_rnn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class attractor():
    # Attractor class is a model level object
    # Since we had used a non-serializable custom rnn layer... we need to rebuild the model using build_model()
    # If the attractor structure change please update build_model()

    def __init__(self, attractor_cfg, h5_name):

        self.cfg = attractor_cfg
        self.build_model()
        self.model.summary()

        self.model.load_weights(self.cfg.path_weight_folder + h5_name)
        rnn_layer = self.model.get_layer('rnn')
        names = [weight.name for weight in rnn_layer.weights]
        weights = self.model.get_weights()

        for name, weight in zip(names, weights):
            if name.endswith('w_pp:0'):
                self.pretrained_w_pp = weight
            if name.endswith('w_pc:0'):
                self.pretrained_w_pc = weight
            if name.endswith('w_cp:0'):
                self.pretrained_w_cp = weight
            if name.endswith('bias_p:0'):
                self.pretrained_bias_p = weight
            if name.endswith('bias_c:0'):
                self.pretrained_bias_c = weight

    def build_model(self):
        clamp_steps = 14
        input_o = Input(shape=(self.cfg.output_dim, ))
        rnn_model = attractor_rnn(self.cfg, clamp_steps)(input_o)
        self.model = Model(input_o, rnn_model)


def arm_attractor(model, attractor):
    # This function will load attractor weights (w_pp, w_pc, w_cp, bias_p, and bias_c) to model

    n_matrices = len(model.get_layer('rnn').weights)
    new_weights = []

    for i in range(n_matrices):
        # Align model and attractor weight matrices by creating new_weights list

        # Get attractor value if weight matrix name match attractor
        if model.get_layer('rnn').weights[i].name.endswith('w_pp:0'):
            new_weights.append(attractor.pretrained_w_pp)

        if model.get_layer('rnn').weights[i].name.endswith('w_pc:0'):
            new_weights.append(attractor.pretrained_w_pc)

        if model.get_layer('rnn').weights[i].name.endswith('w_cp:0'):
            new_weights.append(attractor.pretrained_w_cp)

        if model.get_layer('rnn').weights[i].name.endswith('bias_p:0'):
            new_weights.append(attractor.pretrained_bias_p)

        if model.get_layer('rnn').weights[i].name.endswith('bias_c:0'):
            new_weights.append(attractor.pretrained_bias_c)

        # Fill original value if this slot have not been filled
        if len(new_weights) < i + 1:
            new_weights.append(model.get_weights()[i])

    model.set_weights(new_weights)
    return model


class rnn_no_cleanup(Layer):
    # In plaut version (rnn_v1), input are identical in O-->H path over timestep
    # Use keras copy layer seems more efficient
    def __init__(self, cfg, **kwargs):
        super(rnn_no_cleanup, self).__init__(**kwargs)

        self.cfg = cfg

        self.rnn_activation = activations.get(cfg.rnn_activation)
        self.weight_regularizer = regularizers.l2(cfg.regularizer_const)
        self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=self.cfg.w_initializer, seed=self.cfg.rng_seed)

        self.w_oh = self.add_weight(
            name='w_oh',
            shape=(self.cfg.input_dim, self.cfg.hidden_units),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_hp = self.add_weight(
            name='w_hp',
            shape=(self.cfg.hidden_units, self.cfg.output_dim),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_pp = self.add_weight(
            name='w_pp',
            shape=(self.cfg.output_dim, self.cfg.output_dim),
            initializer=self.w_initializer,
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
            shape=(self.cfg.output_dim, ),
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
                inputs, [self.cfg.input_dim, self.cfg.output_dim], 2
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
            tf.zeros((1, self.cfg.output_dim), dtype=tf.float32)
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
            pp = tf.matmul(self.act_p_list[t - 1], w_pp)
            
#             # Zero diagonal lock
#             pp = tf.matmul(
#                 self.act_p_list[t - 1],
#                 tf.linalg.set_diag(w_pp, tf.zeros(self.cfg.output_dim))
#             )  

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
        return tensor_shape.as_shape([1, cfg.output_dim] + cfg.n_timesteps)

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
        self.w_initializer = tf.random_normal_initializer(mean=0.0, stddev=self.cfg.w_initializer, seed=self.cfg.rng_seed)

        self.w_oh = self.add_weight(
            name='w_oh',
            shape=(self.cfg.input_dim, self.cfg.hidden_units),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_hp = self.add_weight(
            name='w_hp',
            shape=(self.cfg.hidden_units, self.cfg.output_dim),
            initializer=self.w_initializer,
            regularizer=self.weight_regularizer,
            trainable=True
        )

        self.w_pp = self.add_weight(
            name='w_pp',
            shape=(self.cfg.output_dim, self.cfg.output_dim),
            initializer=self.w_initializer,
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
            shape=(self.cfg.output_dim, ),
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
                inputs, [self.cfg.input_dim, self.cfg.output_dim], 2
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
            tf.zeros((1, self.cfg.output_dim), dtype=tf.float32)
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
        return tensor_shape.as_shape([1, cfg.output_dim] + cfg.n_timesteps)

    def get_config(self):
        config = {'custom_cfg': self.cfg, 'name': 'rnn'}
        base_config = super(rnn_pho_task, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ModelCheckpoint_custom(Callback):
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
                
                