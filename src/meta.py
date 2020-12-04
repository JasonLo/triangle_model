import itertools
import json
import os
import uuid

import pandas as pd
import tensorflow as tf
from tqdm import tqdm


from scr.evaluate import vis


def check_gpu():
    if tf.config.experimental.list_physical_devices("GPU"):
        print("GPU is available \n")
    else:
        print("GPU is NOT AVAILABLE \n")


def gpu_mem_cap(b=2048):
    """
    Set GPU memory cap per python kernal for parallel run
    Smaller models usually do not need 100% GPU throughput
    By limiting the memory cap per python kernal, 
    we can run multiple models in parallel to maximize efficiency
    OSP model can archieve 2-3x saving
    """

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=b
                    )
                ],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")

        except:
            pass
    else:
        print("No GPU")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ModelPath:
    def __init__(self, tf_root, code_name, save_freq, total_number_of_epoch):
        # Handle folders
        self.model_folder = os.path.join(tf_root, "models", code_name)
        self.weight_folder = os.path.join(self.model_folder, "weights")
        self.plot_folder = os.path.join(self.model_folder, "plots")
        self.tensorboard_folder = os.path.join(
            tf_root, "tensorboard_log", code_name)

        os.makedirs(self.weight_folder, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)

        # Generate other path related stuffs
        self.weights_checkpoint_fstring = os.path.join(
            self.weight_folder, 'ep{epoch:04d}.h5')
        self.history_pickle = os.path.join(self.model_folder, 'history.pkl')
        self.save_epoches = list(
            range(1, 11)) + list(range(10 + save_freq, total_number_of_epoch + 1, save_freq))
        self.weights_list = [os.path.join(
            self.weight_folder, f"ep{epoch:04d}.h5") for epoch in self.save_epoches]


def make_path(self, tf_root, code_name, save_freq, total_number_of_epoch):

    keys = ("model_folder", "weight_folder", "plot")
    # Handle folders
    self.model_folder = os.path.join(tf_root, "models", code_name)
    self.weight_folder = os.path.join(self.model_folder, "weights")
    self.plot_folder = os.path.join(self.model_folder, "plots")
    self.tensorboard_folder = os.path.join(
        tf_root, "tensorboard_log", code_name)

    os.makedirs(self.weight_folder, exist_ok=True)
    os.makedirs(self.plot_folder, exist_ok=True)

    # Generate other path related stuffs
    self.weights_checkpoint_fstring = os.path.join(
        self.weight_folder, 'ep{epoch:04d}.h5')
    self.history_pickle = os.path.join(self.model_folder, 'history.pkl')
    self.save_epoches = list(
        range(1, 11)) + list(range(10 + save_freq, total_number_of_epoch + 1, save_freq))
    self.weights_list = [os.path.join(
        self.weight_folder, f"ep{epoch:04d}.h5") for epoch in self.save_epoches]


CORE_CONFIGS = ('code_name',
                'tf_root',
                'sample_name',
                'rng_seed',
                'use_semantic',
                'input_dim',
                'hidden_units',
                'output_dim',
                'cleanup_units',
                'pretrain_attractor',
                'tau',
                'max_unit_time',
                'output_ticks',
                'rnn_activation',
                'w_initializer',
                'regularizer_const',
                'p_noise',
                'optimizer',
                'zero_error_radius',
                'n_mil_sample',
                'batch_size',
                'learning_rate',
                'save_freq')

AUX_CONFIGS = ('sampling_speed',
               'sem_param_gf',
               'sem_param_gi',
               'sem_param_kf',
               'sem_param_ki',
               'sem_param_hf',
               'sem_param_hi',
               'embed_attractor_cfg',
               'embed_attractor_h5',
               'w_oh_noise',
               'w_hp_noise',
               'w_pp_noise',
               'w_pc_noise',
               'w_cp_noise',
               'bias_h_noise',
               'bias_c_noise',
               'bias_p_noise',
               'uuid',
               'nEpo',
               'n_timesteps',
               'steps_per_epoch',
               'save_freq_sample',
               'eval_freq',
               'batch_unique_setting_string',
               'show_plots_in_notebook',
               'batch_name',)


class ModelConfig:
    """
    This function keeps all global model configurations
    It will be use in almost every object downsteam, from modelling, evaluation, and visualization

    There are two ways to construct this object
    1) Using a json file path, which contains a cfg dictionary by ModelConfig(json_file)
    2) Using a dictionary by ModelConfig(**dict) 

    Arguements details:
    ------------------------------------------------------------------------------------------------
    >>>META DATA<<<
    code_name: Cfg meta-label, it wont' be use in the model, but it will be recorded in the cfg.json

    >>>TRAINING RELATED<<<
    sample_name: Sampling probability implementation name, see data_wrangling for details
    sampling_speed: Only use in "developmental_rank_frequency" sampling, speed of introducing new words. High = earlier
                    Already adjusted by the no.of sample in the model (cfg.n_mil_sample)
                    See data_wrangling.get_sampling_probability() for details
    rng_seed: Random seed for sampling and tf
    w_initializer: Weight initializer
    regularizer_const: L2 regularization constant (in weight and biases)
    optimizer: Optimizer ('adam' or 'sgd' only)
    learning_rate: Learning rate in optimizer
    n_mil_sample: Stop training after n million sample
    batch_size: Batch size
    save_freq: How often (1 = 10k sample) to save weight after 100k samples. 
               *Model automatically save in every 10k samples in the first 100k sample.

    >>>MODEL ARCHITECHTURE<<<
    use_semantic: To use semantic dummy input or not
        if TRUE, must provide the fomula parameters in the following arguments:
            sem_param_gf, 
            sem_param_gi,
            sem_param_kf,
            sem_param_ki,
            sem_param_hf,
            sem_param_hi
    input_dim: Input dimension
    hidden_units: Number of hidden units in hidden layer
    cleanup_units: Number of cleanup units in attractor network
    pretrain_attractor: A flag to indicate use pretrained attractor or not
        if TRUE: Must provide the pretrianed attractor cfg and weight in the following arguments:
            embed_attractor_cfg',
            embed_attractor_h5',
    output_dim: Output dimension (in one time step)
    rnn_activation: Activation unit use in the recurrent part in the model
    tau: Time averaged input (TAI) parameter tau
    max_unit_time: TAI max unit of time
    output_ticks: How many output ticks should be exported and BPTT from
    p_noise: Gaussian noise in phonolgical system (W_pp, W_pc, W_cp)
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if 'uuid' not in kwargs.keys():
            print("init from scratch")
            self._check_cfg()
            self._store_noise()
            self._init_from_scratch()
            self.path = ModelPath(self.tf_root, self.code_name,
                                  self.save_freq, self.total_number_of_epoch).__dict__

    @classmethod
    def from_json(cls, json_file):
        with open(json_file) as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save(self, json_file):
        self.noise_on()
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f)

        print(f"Saved config json to {json_file}")

    def _init_from_scratch(self):
        # Unique identifier

        self.uuid = uuid.uuid4().hex

        # Additional convienient attributes
        self.n_timesteps = int(self.max_unit_time * (1 / self.tau))
        self.total_number_of_epoch = int(self.n_mil_sample * 1e2)
        self.steps_per_epoch = int(10000 / self.batch_size)

        self.save_freq_sample = self.save_freq * \
            self.batch_size * self.steps_per_epoch  # For TF 2.1
        self.eval_freq = self.save_freq

    def _store_noise(self):
        # Noise management
        self.w_pp_noise = self.p_noise
        self.w_pc_noise = self.p_noise
        self.w_cp_noise = self.p_noise
        self.w_oh_noise = 0.
        self.w_hp_noise = 0.
        self.bias_h_noise = 0.
        self.bias_c_noise = self.p_noise
        self.bias_p_noise = self.p_noise

        self.w_oh_noise_backup = self.w_oh_noise
        self.w_hp_noise_backup = self.w_hp_noise
        self.w_pp_noise_backup = self.w_pp_noise
        self.w_pc_noise_backup = self.w_pc_noise
        self.w_cp_noise_backup = self.w_cp_noise
        self.bias_h_noise_backup = self.bias_h_noise
        self.bias_c_noise_backup = self.bias_c_noise
        self.bias_p_noise_backup = self.bias_p_noise

    def _check_cfg(self):
        # Check all ingested_keys fufill minimal cfg requirement
        if not all([x in vars(self) for x in CORE_CONFIGS]):
            raise ValueError(
                'Some cfg is undefined, double check cfg contains all necessary params')

        if self.sample_name == "developmental_rank_frequency":
            assert type(self.sampling_speed) == float

        if self.pretrain_attractor:
            assert type(self.embed_attractor_cfg) == str
            assert type(self.embed_attractor_h5) == str

    def noise_on(self):
        # Noise is on by default
        self.w_oh_noise = self.w_oh_noise_backup
        self.w_hp_noise = self.w_hp_noise_backup
        self.w_pp_noise = self.w_pp_noise_backup
        self.w_pc_noise = self.w_pc_noise_backup
        self.w_cp_noise = self.w_cp_noise_backup
        self.bias_h_noise = self.bias_h_noise_backup
        self.bias_c_noise = self.bias_c_noise_backup
        self.bias_p_noise = self.bias_p_noise_backup

    def noise_off(self):
        self.w_oh_noise = 0.
        self.w_hp_noise = 0.
        self.w_pp_noise = 0.
        self.w_pc_noise = 0.
        self.w_cp_noise = 0.
        self.bias_h_noise = 0.
        self.bias_c_noise = 0.
        self.bias_p_noise = 0.


def make_batch_cfg(batch_name, batch_output_dir, static_hpar, param_grid, in_notebook):
    """
    Make batch cfg dictionary list that can feed into papermill
    """

    # Check batch json exist
    batch_json = batch_output_dir + "batch_config.json"

    if os.path.isfile(batch_json):
        print("Batch config json is found, load cfgs from disk")
        with open(batch_json) as f:
            batch_cfgs = json.load(f)
    else:
        # Make batch_cfgs from given parameters

        # Check duplicate keys
        for key in static_hpar.keys():
            if key in param_grid.keys():
                raise ValueError(
                    "Key duplicated in vary and static parameter: {}".format(key))

        # Iterate and create batch level super object: batch_cfgs
        batch_cfgs = []
        varying_hpar_names, varying_hpar_values = zip(*param_grid.items())
        for i, v in enumerate(itertools.product(*varying_hpar_values)):
            code_name = batch_name + "_r{:04d}".format(i)

            this_hpar = dict(zip(varying_hpar_names, v))
            this_hpar.update(static_hpar)

            # Add identifier params into param dict
            this_hpar["code_name"] = code_name

            setting_list_without_rng_seed = [
                x + str(this_hpar[x]) for x in varying_hpar_names if x != 'rng_seed'
            ]
            this_hpar["batch_unique_setting_string"] = '_'.join(
                setting_list_without_rng_seed)

            # Pass into ModelConfig to catch error early
            ModelConfig(**this_hpar, just_chk=True)

            batch_cfg = dict(
                sn=i,
                in_notebook=in_notebook,
                code_name=code_name,
                model_folder="models/" + code_name + "/",
                out_notebook="models/" + code_name + "/output.ipynb",
                params=this_hpar,
            )

            batch_cfgs.append(batch_cfg)

        # Save batch cfg to json
        os.makedirs(batch_output_dir, exist_ok=True)
        with open(batch_json, "w") as f:
            json.dump(batch_cfgs, f)

        print("Batch config saved to {}".format(batch_output_dir))

    print("There are {} models in this batch".format(len(batch_cfgs)))

    return batch_cfgs


def parse_batch_results(cfgs):
    """
    Parse and Concat all condition level results from item level csvs
    And merge with cfg data (run level meta data) from cfgs
    cfgs: batch cfgs in dictionary format (The one we saved to disk, for running papermill)
    """

    evals_df = pd.DataFrame()
    cfgs_df = pd.DataFrame()

    for i in tqdm(range(len(cfgs))):

        # Extra cfg (with UUID) from actual saved cfg json
        this_ModelConfig = ModelConfig(
            cfgs[i]['model_folder'] + 'model_config.json'
        )
        cfgs_df = pd.concat(
            [cfgs_df,
             pd.DataFrame(this_ModelConfig.to_dict(), index=[i])]
        )

        # Evaluate results
        this_eval = vis(cfgs[i]['model_folder'])
        this_eval.parse_cond_df()
        evals_df = pd.concat([evals_df, this_eval.cdf], ignore_index=True)

    return cfgs_df, pd.merge(evals_df, cfgs_df, 'left', 'code_name')
