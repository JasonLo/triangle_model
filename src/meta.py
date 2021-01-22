import itertools
import json
import os
import uuid
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from evaluate import vis


# Parameters registry
CORE_CONFIGS = (
    "code_name",
    "tf_root",
    "ort_units",
    "pho_units",
    "sem_units",
    "hidden_os_units",
    "hidden_op_units",
    "hidden_ps_units",
    "hidden_sp_units",
    "pho_cleanup_units",
    "sem_cleanup_units",
    "pho_noise_level",
    "sem_noise_level",
    "activation",
    "tau",
    "max_unit_time",
    "output_ticks",
    "sample_name",
    "rng_seed",
    "learning_rate",
    "n_mil_sample",
    "batch_size",
    "save_freq",
)

OPTIONAL_CONFIGS = (
    "sampling_speed",
    "batch_name",
    "batch_unique_setting_string",
    "oral_vocab_size"
)


class ModelConfig:
    """This function keeps all global model configurations
    It will be use in almost every object downsteam, from modelling, evaluation, and visualization
    There are two ways to construct this object
    1) Using a json file path, which contains a cfg dictionary by ModelConfig(json_file)
    2) Using a dictionary by ModelConfig(**dict)
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "uuid" not in kwargs.keys():
            print("init from scratch")
            self._init_from_scratch()

    @classmethod
    def from_json(cls, json_file):
        """Create ModelConfig from json file"""
        print(f"Loading config from {json_file}")
        with open(json_file) as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)

    def __call__(self):
        return self.__dict__


    def _init_from_scratch(self):

        self._pho_noise_level_backup = self.pho_noise_level

        # Unique identifier
        self.uuid = uuid.uuid4().hex

        # Additional convienient attributes
        self.n_timesteps = int(self.max_unit_time * (1 / self.tau))
        self.total_number_of_epoch = int(self.n_mil_sample * 1e2)
        self.steps_per_epoch = int(10000 / self.batch_size)

        self.save_freq_sample = (
            self.save_freq * self.batch_size * self.steps_per_epoch
        )  # For TF 2.1
        self.eval_freq = self.save_freq

        self.saved_epoches = list(range(1, 11)) + list(
            range(10 + self.save_freq, self.total_number_of_epoch + 1, self.save_freq)
        )
        self.path = self._make_path()

    def _make_path(self):
        path_dict = {}
        path_dict["tf_root"] = self.tf_root
        path_dict["model_folder"] = os.path.join(self.tf_root, "models", self.code_name)
        path_dict["weight_folder"] = os.path.join(path_dict["model_folder"], "weights")
        path_dict["save_model_folder"] = os.path.join(path_dict["model_folder"], "saved_model")
        path_dict["plot_folder"] = os.path.join(path_dict["model_folder"], "plots")
        path_dict["tensorboard_folder"] = os.path.join(self.tf_root, "tensorboard_log", self.code_name)
        path_dict["weights_checkpoint_fstring"] = os.path.join(path_dict["weight_folder"], "ep{epoch:04d}")
        path_dict["history_pickle"] = os.path.join(path_dict["model_folder"], "history.pkl")
        path_dict["weights_list"] = [
            os.path.join(path_dict["weight_folder"], f"ep{epoch:04d}")
            for epoch in self.saved_epoches
        ]

        os.makedirs(path_dict["weight_folder"], exist_ok=True)
        os.makedirs(path_dict["plot_folder"], exist_ok=True)
        os.makedirs(path_dict["save_model_folder"], exist_ok=True)
        return path_dict

    def noise_on(self):
        # Noise is on by default
        self.pho_noise_level = self._pho_noise_level_backup

    def noise_off(self):
        self.pho_noise_level = 0.0

    def save(self, json_file=None):
        self.noise_on()

        if json_file is None:
            json_file = os.path.join(self.path["model_folder"], "model_config.json")
            
        with open(json_file, "w") as f:
            json.dump(self.__dict__, f)
        print(f"Saved config json to {json_file}")


# %% Batch related functions
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
                    "Key duplicated in vary and static parameter: {}".format(key)
                )

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
                x + str(this_hpar[x]) for x in varying_hpar_names if x != "rng_seed"
            ]
            this_hpar["batch_unique_setting_string"] = "_".join(
                setting_list_without_rng_seed
            )

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
        this_ModelConfig = ModelConfig(cfgs[i]["model_folder"] + "model_config.json")
        cfgs_df = pd.concat(
            [cfgs_df, pd.DataFrame(this_ModelConfig.to_dict(), index=[i])]
        )

        # Evaluate results
        this_eval = vis(cfgs[i]["model_folder"])
        this_eval.parse_cond_df()
        evals_df = pd.concat([evals_df, this_eval.cdf], ignore_index=True)

    return cfgs_df, pd.merge(evals_df, cfgs_df, "left", "code_name")


# %% Other misc functions
def check_gpu():
    """Check whether GPU available"""
    if tf.config.experimental.list_physical_devices("GPU"):
        print("GPU is available \n")
    else:
        print("GPU is NOT AVAILABLE \n")


def set_gpu_mem_cap(b=2048):
    """
    Set GPU memory cap per python kernal for parallel run
    Smaller models usually do not need 100% GPU throughput
    By limiting the memory cap per python kernal,
    we can run multiple models in parallel to maximize efficiency
    OSP model can archieve 2-3x saving.

    Use nvidia-smi in terminal to check total available memory
    """

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=b)],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")

        except:
            pass
    else:
        print("No GPU")