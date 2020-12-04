import itertools
import json
import os

import pandas as pd
import pandas_gbq
import tensorflow as tf


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
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
            )

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


class model_cfg:
    """
    This function keeps all global model configurations
    It will be use in almost every object downsteam, from modelling, evaluation, and visualization

    There are two ways to construct this object
    1) Using a json file path, which contains a cfg dictionary by model_cfg(json_file)
    2) Using a dictionary by model_cfg(**dict) 

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
    minimal_cfgs = ['code_name',
                    'path_tf_root',
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
                    'save_freq']

    aux_cfgs = [
        'sampling_speed',
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
        'batch_name',
    ]

    tmp_cfgs = ['w_oh_noise_backup',
                'w_hp_noise_backup',
                'w_pp_noise_backup',
                'w_pc_noise_backup',
                'w_cp_noise_backup',
                'bias_h_noise_backup',
                'bias_c_noise_backup',
                'bias_p_noise_backup',
                'path_weights_checkpoint',
                'path_weights_list',
                'path_plot_folder',
                'path_weight_folder',
                'path_history_pickle',
                'saved_epoch_list',
                ]

    all_cfgs_name = minimal_cfgs + aux_cfgs + tmp_cfgs

    def __init__(self, json_file=None, bypass_chk=False, just_chk=False, **kwargs):
        # Validate json file
        if type(json_file) == str and json_file.endswith('.json'):
            with open(json_file) as f:
                kwargs = json.load(f)

        # Set attributes from json values
        invalid_keys = []
        for key, value in kwargs.items():
            if key in self.all_cfgs_name:
                setattr(self, key, value)
            else:
                invalid_keys.append(key)

        if len(invalid_keys) > 0:
            print('These keys in cfg file is not valid: {}'.format(invalid_keys))

        # Additional initialization for dictionary constructor
        if json_file == None:
            self.init_from_dict()

        if not bypass_chk:
            self.chk_cfg()

        if not just_chk:
            self.store_noise()
            self.gen_paths()

            if (json_file == None):
                self.write_cfg_to_json()

    def init_from_dict(self):
        # Unique identifier
        import uuid
        self.uuid = uuid.uuid4().hex

        # Additional convienient attributes
        self.n_timesteps = int(self.max_unit_time * (1 / self.tau))
        self.nEpo = int(self.n_mil_sample * 1e2)
        self.steps_per_epoch = int(10000 / self.batch_size)

        # Saving cfg
        self.save_freq_sample = self.save_freq * \
            self.batch_size * self.steps_per_epoch  # For TF 2.1
        self.eval_freq = self.save_freq

    def to_dict(self):
        """
        Get a trimed dictionary (dropped attribute in tmp_cfg)
        """
        return {key: getattr(self, key) for key in (self.minimal_cfgs + self.aux_cfgs)}

    def __str__(self):
        return str(self.to_dict())

    def store_noise(self):
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

    def chk_cfg(self):
        # Check all ingested_keys fufill minimal cfg requirement
        if not all([x in vars(self) for x in self.minimal_cfgs]):
            raise ValueError(
                'Some cfg is undefined, double check cfg contains all necessary params')

        if self.sample_name == "developmental_rank_frequency":
            assert type(self.sampling_speed) == float

        if self.use_semantic == True:
            if not (type(self.sem_param_gf) == float):
                raise ValueError('check sem_params')
            if not (type(self.sem_param_gi) == float):
                raise ValueError('check sem_params')
            if not (type(self.sem_param_kf) == float):
                raise ValueError('check sem_params')
            if not (type(self.sem_param_ki) == float):
                raise ValueError('check sem_params')
            if not (type(self.sem_param_hf) == float):
                raise ValueError('check sem_params')
            if not (type(self.sem_param_hi) == float):
                raise ValueError('check sem_params')
        else:
            self.sem_param_gf = None
            self.sem_param_gi = None
            self.sem_param_kf = None
            self.sem_param_ki = None
            self.sem_param_hf = None
            self.sem_param_hi = None

        if self.pretrain_attractor == True:
            if not (type(self.embed_attractor_cfg) == str):
                raise ValueError('check embed_attractor_cfg')
            if not (type(self.embed_attractor_h5) == str):
                raise ValueError('check embed_attractor_h5')
        else:
            self.embed_attractor_cfg = None
            self.embed_attractor_h5 = None

    def gen_paths(self):
        self.path_model_folder = self.path_tf_root + '/models/' + self.code_name + '/'
        self.path_weight_folder = self.path_model_folder + 'weights/'
        self.path_plot_folder = self.path_model_folder + 'plots/'

        os.makedirs(self.path_weight_folder, exist_ok=True)
        os.makedirs(self.path_plot_folder, exist_ok=True)

        self.path_weights_checkpoint = self.path_weight_folder + \
            'ep{epoch:04d}.h5'
        self.path_history_pickle = self.path_model_folder + 'history.pickle'
        self.path_weights_list = []
        self.saved_epoch_list = []

        for epoch in range(1, 11):
            self.path_weights_list.append(
                self.path_weight_folder + 'ep' + str(epoch).zfill(4) + '.h5')
            self.saved_epoch_list.append(epoch)

        for epoch in range(10+self.save_freq, self.nEpo + 1, self.save_freq):
            self.path_weights_list.append(
                self.path_weight_folder + 'ep' + str(epoch).zfill(4) + '.h5')
            self.saved_epoch_list.append(epoch)

    def noise_off(self):
        self.w_oh_noise = 0.
        self.w_hp_noise = 0.
        self.w_pp_noise = 0.
        self.w_pc_noise = 0.
        self.w_cp_noise = 0.
        self.bias_h_noise = 0.
        self.bias_c_noise = 0.
        self.bias_p_noise = 0.

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

    def write_cfg_to_json(self):

        if os.path.isfile(self.path_model_folder + 'model_config.json'):
            print('='*50)
            print(
                'Found model_config.json on disk, I will NEVER overwrite it automatically.')
            print('Manually delete config if you are sure.')
            print('Or save this model into another folder by changing cfg.code_name')
            print('='*50)

        else:
            # Make sure noise is armed before saving, since loading will copy noise to backup
            self.noise_on()
            save_cfg = {k: vars(self)[k] for k in self.all_cfgs_name}
            with open(self.path_model_folder + 'model_config.json', 'w') as f:
                json.dump(save_cfg, f)


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

            # Pass into model_cfg to catch error early
            model_cfg(**this_hpar, just_chk=True)

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
    from src.evaluate import vis
    from tqdm import tqdm
    """
    Parse and Concat all condition level results from item level csvs
    And merge with cfg data (run level meta data) from cfgs
    cfgs: batch cfgs in dictionary format (The one we saved to disk, for running papermill)
    """

    evals_df = pd.DataFrame()
    cfgs_df = pd.DataFrame()

    for i in tqdm(range(len(cfgs))):

        # Extra cfg (with UUID) from actual saved cfg json
        this_model_cfg = model_cfg(
            cfgs[i]['model_folder'] + 'model_config.json'
        )
        cfgs_df = pd.concat(
            [cfgs_df,
             pd.DataFrame(this_model_cfg.to_dict(), index=[i])]
        )

        # Evaluate results
        this_eval = vis(cfgs[i]['model_folder'])
        this_eval.parse_cond_df()
        evals_df = pd.concat([evals_df, this_eval.cdf], ignore_index=True)

    return cfgs_df, pd.merge(evals_df, cfgs_df, 'left', 'code_name')


class connect_gbq:
    """
    All things related to GBQ
    """

    def __init__(self, pid='triangle-272405', credential_json="triangle-e1fd21bb86a1.json"):
        from google.oauth2 import service_account
        self.pid = pid
        self.credentials = service_account.Credentials.from_service_account_file(
            credential_json
        )

    def push_all(self, db_name, cfg, strain_i_hist, grain_i_hist, verbose=False):
        """
        Push a single run to GBQ including:
        - cfg
        - Strain item history
        - Grain item history
        """

        if verbose:
            print('Writing data to Bigquery')

        # Config file
        pandas_gbq.to_gbq(
            pd.DataFrame(cfg, index=[0]),
            destination_table=db_name + '.cfg',
            project_id=self.pid,
            if_exists='append',
            credentials=self.credentials,
            progress_bar=False
        )

        # Strain eval
        pandas_gbq.to_gbq(
            strain_i_hist,
            destination_table=db_name + '.strain',
            project_id=self.pid,
            if_exists='append',
            credentials=self.credentials,
            progress_bar=False
        )

        # Grain eval
        pandas_gbq.to_gbq(
            grain_i_hist,
            destination_table=db_name + '.grain',
            project_id=self.pid,
            if_exists='append',
            credentials=self.credentials,
            progress_bar=False
        )

        if verbose:
            print('Completed')

    def read_bq_cfg(self, db_name):
        """
        Read cfg from GBQ
        """
        sql = """
        SELECT * FROM `{}.{}.cfg`
        """.format(self.pid, db_name)
        return pandas_gbq.read_gbq(
            sql, project_id=self.pid, credentials=self.credentials
        )


def send_mail(batch_name, email='lcmjlo@gmail.com'):
    """
    Send an email to myself for super long run
    """
    import smtplib
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('gcpdazzo@gmail.com', 'l3gnonveppl#')

    subject = 'Batch training {batch_name} completed'
    body = 'Job done'
    msg = 'Subject: {}\n\n{}'.format(subject, body)

    server.sendmail('gcpdazzo@gmail.com', email, msg)
    print('Email sent')
    server.quit()
