import itertools
import json
import os
import uuid
import tensorflow as tf
from dataclasses import dataclass

# Use dataclass to replace manual config registry.
# # Parameters registry
# CORE_CONFIGS = (
#     "code_name",
#     "tf_root",
#     "ort_units",
#     "pho_units",
#     "sem_units",
#     "hidden_os_units",
#     "hidden_op_units",
#     "hidden_ps_units",
#     "hidden_sp_units",
#     "pho_cleanup_units",
#     "sem_cleanup_units",
#     "pho_noise_level",
#     "sem_noise_level",
#     "activation",
#     "tau",
#     "max_unit_time",
#     "inject_error_ticks",
#     "output_ticks",
#     "learning_rate",
#     "zero_error_radius",
#     "save_freq",
# )

# ENV_CONFIGS = (
#     "tasks",
#     "wf_compression",
#     "wf_clip_low",
#     "wf_clip_high",
#     "oral_start_pct",
#     "oral_end_pct",
#     "oral_sample",
#     "oral_tasks_ps",
#     "transition_sample",
#     "reading_sample",
#     "reading_tasks_ps",
#     "batch_size",
#     "rng_seed",
# )

# OPTIONAL_CONFIGS = (
#     "batch_name",
#     "batch_unique_setting_string",
#     "pretrained_checkpoint",
# )


@dataclass
class ModelConfig:
    code_name: str
    tf_root: str = "/home/jupyter/tf"
    uuid: str = None
    batch_name: str = None
    batch_unique_setting_string: str = None

    # Model configs
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

    tau: float = 1 / 3
    max_unit_time: float = 4.0
    output_ticks: int = 11
    inject_error_ticks: int = 11

    # Training
    learning_rate: float = 0.005
    zero_error_radius: float = None
    save_freq: int = 10

    # Environment
    tasks: tuple = ("pho_sem", "sem_pho", "pho_pho", "sem_sem", "triangle")
    wf_compression: str = "log"
    wf_clip_low: int = 0
    wf_clip_high: int = 999_999_999
    oral_start_pct: float = 0.02
    oral_end_pct: float = 0.5
    oral_sample: int = 900_000
    oral_tasks_ps: tuple = (0.4, 0.4, 0.1, 0.1, 0.0)
    transition_sample: int = 400_000
    reading_sample: int = 3_100_000
    reading_tasks_ps: tuple = (0.2, 0.2, 0.05, 0.05, 0.5)
    batch_size: int = 100
    rng_seed: int = 2021

    def __post_init__(self):
        os.makedirs(self.weight_folder, exist_ok=True)
        os.makedirs(self.eval_folder, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)

        if self.uuid is None:
            print("UUID not found, regenerating.")
            self.uuid = uuid.uuid4().hex
            self.save()

    @classmethod
    def from_json(cls, json_file):
        """Create ModelConfig from json file"""
        print(f"Loaded config from {json_file}")
        with open(json_file) as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    @classmethod
    def from_global(cls, globals_dict):
        config_dict = {k: globals_dict[k] for k in globals_dict if k in cls.__annotations__.keys()}
        return cls(**config_dict)

    def save(self, json_file=None):
        """Save run config to json file"""
        if json_file is None:
            json_file = os.path.join(self.model_folder, "model_config.json")

        with open(json_file, "w") as f:
            json.dump(self.__dict__, f)
            
        print(f"Saved config json to {json_file}")

    # Inferred training time properties
    @property
    def n_timesteps(self) -> int:
        return int(self.max_unit_time * (1 / self.tau))

    @property
    def total_sample(self) -> int:
        return self.oral_sample + self.reading_sample

    @property
    def total_number_of_epoch(self) -> int:
        return int(self.total_sample / 10000)
    
    @property
    def steps_per_epoch(self) -> int:
        return int(10000 / self.batch_size)

    @property
    def saved_epoches(self) -> list:
        oral_first_10 = list(range(1, 11))
        first_read_epoch = int((self.oral_sample / 10000) + 1)
        read_first_10 = list(range(first_read_epoch, first_read_epoch + 10))
        other = list(range(self.save_freq, self.total_number_of_epoch + 1, self.save_freq))
        return sorted(list(set(oral_first_10 + read_first_10 + other)))

    # Path related config properties
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
        return os.path.join(self.weight_folder, "ep{epoch:04d}")

    @property
    def saved_weights(self) -> list:
        return [os.path.join(self.weight_folder, f"ep{epoch:04d}") for epoch in self.saved_epoches]

    @property
    def config_json(self) -> str:
        return os.path.join(self.model_folder, 'config.json')



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


# %% Other misc functions
def check_gpu():
    """Check whether GPU available"""
    if tf.config.experimental.list_physical_devices("GPU"):
        print("GPU is available \n")
    else:
        print("GPU is NOT AVAILABLE \n")


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


def batch_config_to_bigquery(batch_cfgs_json, dataset_name, table_name):
    from google.cloud import bigquery
    import json, os
    import pandas as pd

    with open(batch_cfgs_json) as f:
        batch_cfgs = json.load(f)

    df = pd.DataFrame()
    for i, cfg in enumerate(batch_cfgs):
        # get_uuid from saved model_json
        model_config_json = os.path.join(
            cfg["params"]["tf_root"], cfg["model_folder"], "model_config.json"
        )
        with open(model_config_json) as f:
            model_config = json.load(f)

        # Copy uuid from model config to batch config
        cfg["params"]["uuid"] = model_config["uuid"]

        # Gather config to a dataframe
        df = pd.concat([df, pd.DataFrame(cfg["params"], index=[i])])

    # Create connection to BQ and push data
    client = bigquery.Client()
    dataset = client.create_dataset(dataset_name, exists_ok=True)
    table_ref = dataset.table(table_name)
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print("Loaded dataframe to {}".format(table_ref.path))


def csv_to_bigquery(csv_file, dataset_name, table_name):
    from google.cloud import bigquery
    import json, os
    import pandas as pd

    # Create connection to BQ and push data
    client = bigquery.Client()
    dataset = client.create_dataset(dataset_name, exists_ok=True)
    table_ref = dataset.table(table_name)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True
    )

    with open(csv_file, "rb") as f:
        job = client.load_table_from_file(f, table_ref, job_config=job_config)

    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_name}:{table_ref.path}")
