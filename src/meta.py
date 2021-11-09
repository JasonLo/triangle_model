import itertools
import json
import os
import uuid
import tensorflow as tf
from dataclasses import dataclass
from environment import EnvironmentConfig
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


@dataclass
class ModelConfig:
    """ModelConfig Class contains all the information about the model architecture and training.

    Keyword arguments:
        ort_units -- number of orthographic units
        pho_units -- number of phonological units
        sem_units -- number of semantic units
        hidden_os_units -- number of hidden units in O to S
        hidden_op_units -- number of hidden units in O to P
        hidden_ps_units -- number of hidden units in P to S
        hidden_sp_units -- number of hidden units in S to P
        pho_clean_units -- number of cleanup (hidden) units in phonology attractor
        sem_clean_units -- number of cleanup (hidden) units in semantic attractor
        pho_noise_level -- Gaussian noise level at phonology system measured in SD (also see modeling._inject_noise_to_all_pho())
        sem_noise_level -- Gaussian noise level at semantic system measured in SD (also see modeling._inject_noise_to_all_sem())
        activation -- name of activation function for all layers
        tau -- time constant in time averaged input
        max_unit_time -- maximum unit time in time averaged input
        output_ticks -- number of ticks to output
        inject_error_ticks -- number of ticks to inject error, start from last ticks
        learning_rate -- learning rate for optimizer
        zero_error_radius -- whether to use zero error radius or not and if so, what radius to use (e.g., None or 0.1)
    """

    # Architecture
    ort_units: int = 119
    pho_units: int = 250
    sem_units: int = 2446
    hidden_os_units: int = 500
    hidden_op_units: int = 100
    hidden_ps_units: int = 500
    hidden_sp_units: int = 500
    pho_cleanup_units: int = 50
    sem_cleanup_units: int = 50
    pho_noise_level: float = 0.0
    sem_noise_level: float = 0.0
    activation: str = "sigmoid"

    # Time related
    tau: float = 1 / 3
    max_unit_time: float = 4.0
    output_ticks: int = 11
    inject_error_ticks: int = 11

    # Training related
    learning_rate: float = 0.005
    zero_error_radius: float = None

    @classmethod
    def from_dict(cls, **kwargs):
        """Create a ModelConfig from a dictionary."""
        config_dict = {
            k: v for k, v in kwargs.items() if k in cls.__annotations__.keys()
        }
        return cls(**config_dict)


@dataclass
class Config:
    """Global configuration with training environment, model config, and other misc configurations."""

    code_name: str
    model_config: ModelConfig
    environment_config: EnvironmentConfig
    rng_seed: int = 2021
    save_freq: int = 10
    uuid: str = None
    batch_name: str = None
    batch_unique_setting_string: str = None

    def __post_init__(self):
        # Elevate subsidary configs to class level
        self.__dict__.update(self.model_config.__dict__)
        self.__dict__.update(self.environment_config.__dict__)

        # Create folders
        all_folders = [
            self.model_folder,
            self.weight_folder,
            self.eval_folder,
            self.plot_folder,
            self.tensorboard_folder,
        ]
        [os.makedirs(f, exist_ok=True) for f in all_folders]

        # Create unique string for this run
        if self.uuid is None:
            print("UUID not found, regenerating.")
            self.uuid = uuid.uuid4().hex
            self.save()  # Only save if created from scratch (without uuid)

    @classmethod
    def from_dict(cls, **kwargs):
        """Create Config from all available keys, including base, env, and model."""

        base_config = {k: kwargs[k] for k in kwargs if k in cls.__annotations__.keys()}

        model_config_keys = ModelConfig.__annotations__.keys()
        model_config = ModelConfig(**{k: kwargs[k] for k in model_config_keys})

        environment_config_keys = EnvironmentConfig.__annotations__.keys()
        environment_config = EnvironmentConfig(
            **{k: kwargs[k] for k in environment_config_keys}
        )

        return cls(
            model_config=model_config,
            environment_config=environment_config,
            **base_config,
        )

    @classmethod
    def from_json(cls, json_file):
        """Create ModelConfig from json file"""
        print(f"Loaded config from {json_file}")
        with open(json_file) as f:
            config_dict = json.load(f)

        return cls.from_dict(**config_dict)

    # Path related config properties

    @property
    def tf_root(self) -> str:
        """Return the root folder for tensorflow models."""
        return os.environ.get("TF_ROOT")

    @property
    def model_folder(self) -> str:
        if self.batch_name is not None:
            return os.path.join(self.tf_root, "models", self.batch_name, self.code_name)
        else:
            return os.path.join(self.tf_root, "models", self.code_name)

    @property
    def weight_folder(self) -> str:
        return os.path.join(self.model_folder, "weights")

    @property
    def checkpoint_folder(self) -> str:
        return os.path.join(self.model_folder, "checkpoints")

    @property
    def eval_folder(self) -> str:
        return os.path.join(self.model_folder, "eval")

    @property
    def plot_folder(self) -> str:
        return os.path.join(self.model_folder, "plots")

    @property
    def tensorboard_folder(self) -> str:
        return os.path.join(self.tf_root, "tensorboard_log", self.code_name)

    @property
    def saved_weights_fstring(self) -> str:
        return os.path.join(self.weight_folder, "epoch-{epoch}")

    @property
    def saved_checkpoints_fstring(self) -> str:
        return os.path.join(self.checkpoint_folder, "epoch-{epoch}")

    @property
    def saved_weights(self) -> list:
        return [
            os.path.join(self.weight_folder, self.saved_weights_fstring.format(epoch))
            for epoch in self.saved_epochs
        ]

    @property
    def config_json(self) -> str:
        return os.path.join(self.model_folder, "config.json")

    @property
    def n_timesteps(self) -> int:
        return int(self.max_unit_time * (1 / self.tau))

    @property
    def steps_per_epoch(self) -> int:
        return int(10000 / self.batch_size)

    @property
    def total_number_of_epoch(self) -> int:
        return int(self.total_sample / 10000)

    @property
    def saved_epochs(self) -> list:
        """Saved epoch. 
        Save is more frequent in earlier epoch
        """
        a = list(range(1, 11)) 
        b = list(range(12, 31, 2)) 
        c = list(range(35, 51, 5)) 
        d = [i for i in range(1, self.total_number_of_epoch + 1) if (i % self.save_freq == 0)]
        return sorted(list(set(a + b + c + d)))

    @property
    def papermill_cfg(self) -> dict:
        """A serializeable config dictionary for Papermill."""
        cfg = self.__dict__.copy()
        # Get rid of non-serializable objects
        cfg.pop("model_config")
        cfg.pop("environment_config")
        return cfg

    def save(self, json_file: str = None):
        """Save run config to json file."""

        if json_file is None:
            json_file = os.path.join(
                self.model_folder, "model_config.json"
            )  # default save location

        # Save config
        with open(json_file, "w") as f:
            json.dump(self.papermill_cfg, f)
        print(f"Saved config json to {json_file}")

# Batch related
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

        # Check no duplicate keys
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
            Config.from_dict(**this_hpar)

            batch_cfg = dict(
                sn=i,
                in_notebook=in_notebook,
                code_name=code_name,
                model_folder=os.path.join(batch_output_dir, code_name),
                out_notebook=os.path.join(batch_output_dir, code_name, "output.ipynb"),
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

def batch_json_to_df(json_file:str) -> pd.DataFrame:
    """
    Convert batch json to dataframe
    """
    from dotenv import load_dotenv
    load_dotenv()
    tf_root = os.environ.get("TF_ROOT")

    # Load json
    with open(json_file) as f:
        batch_cfgs = json.load(f)
    
    df = pd.DataFrame()

    for i, cfg in enumerate(batch_cfgs):

        # get_uuid from saved model_json
        model_config_json = os.path.join(
            tf_root, cfg["model_folder"], "model_config.json"
        )

        # Copy uuid from model config to batch config
        with open(model_config_json) as f:
            model_config = json.load(f)
        cfg["params"]["uuid"] = model_config["uuid"]

        # Remove tuples
        cfg["params"] = {k: v for k, v in cfg["params"].items() if type(v) is not list}

        # Stitch
        this_record = pd.DataFrame(cfg["params"], index=[i])
        df = df.append(this_record)

    return df

# Broken
# def parse_batch_results(cfgs):
#     """
#     Parse and Concat all condition level results from item level csvs
#     And merge with cfg data (run level meta data) from cfgs
#     cfgs: batch cfgs in dictionary format (The one we saved to disk, for running papermill)
#     """

#     evals_df = pd.DataFrame()
#     cfgs_df = pd.DataFrame()

#     for i in tqdm(range(len(cfgs))):

#         # Extra cfg (with UUID) from actual saved cfg json
#         this_ModelConfig = ModelConfig(cfgs[i]["model_folder"] + "model_config.json")
#         cfgs_df = pd.concat(
#             [cfgs_df, pd.DataFrame(this_ModelConfig.to_dict(), index=[i])]
#         )

#         # Evaluate results
#         this_eval = vis(cfgs[i]["model_folder"])
#         this_eval.parse_cond_df()
#         evals_df = pd.concat([evals_df, this_eval.cdf], ignore_index=True)

#     return cfgs_df, pd.merge(evals_df, cfgs_df, "left", "code_name")


# GPU related

def check_gpu():
    """Check whether GPU available"""
    if tf.config.experimental.list_physical_devices("GPU"):
        print("GPU is available \n")
    else:
        print("GPU is NOT AVAILABLE \n")


def split_gpu(which_gpu: int, n_splits: int = 2):
    """Split a physical GPU into multiple logical GPUs.
    Some of the models does't require the whole GPU to run, so we can split it into multiple logical GPUs and parallelize on it.
    IMPORTANT: Do not import tensorflow before this function.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    memory_size = int(10000 / n_splits)  # Titan X on Uconn server
    logical_gpus = [
        tf.config.LogicalDeviceConfiguration(memory_limit=memory_size)
        for _ in range(n_splits)
    ]

    try:
        # Use only selected GPU(s)
        tf.config.set_visible_devices(gpus[which_gpu], "GPU")
        tf.config.set_logical_device_configuration(gpus[which_gpu], logical_gpus)

    except Exception:
        # Invalid device or cannot modify virtual devices once initialized.
        raise Exception
        print('Virtual GPU cannot be created')


def limit_gpu_memory_use(limit_MB=7168):
    """
    Set GPU memory cap per python kernal for parallel run
    Smaller models usually do not need 100% GPU throughput
    By limiting the memory cap per python kernal,
    we can run multiple models in parallel to maximize efficiency
    OSP model can get 2-3x saving.

    Use nvidia-smi in terminal to check total available memory
    """
    gpus = tf.config.list_physical_devices("GPU")
    cfg = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit_MB)]

    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], cfg)
    except:
        pass

