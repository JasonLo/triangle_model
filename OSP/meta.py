from collections import OrderedDict
import tensorflow as tf
import altair as alt
import pandas as pd
import pandas_gbq
import json, os

def check_gpu():
    if tf.config.experimental.list_physical_devices("GPU"):
        print("GPU is available \n")
    else:
        print("GPU is NOT AVAILABLE \n")

def gpu_mem_cap(b=2048):
    # Set GPU memory cap per python kernal for parallel run
    # Smaller models usually do not need 100% GPU throughput
    # By limiting the memory cap per python kernal, we can run multiple models in parallel to maximize efficiency
    # OSP model can archieve 2-3x saving

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
    This function keep all global model configurations
    It will be use in almost every object downsteam, from modelling, evaluation, and visualization

    There are two ways to construct this object
    1) Using a json file path, which contains a cfg dictionary
    2) Using a dictionary by model_cfg(**dict) 
    """
    minimal_cfgs = ['code_name',
                    'sample_name',
                    'sample_rng_seed',
                    'tf_rng_seed',
                    'use_semantic',
                    'input_dim',
                    'hidden_units',
                    'output_dim',
                    'cleanup_units',
                    'use_attractor',
                    'tau',
                    'max_unit_time',
                    'rnn_activation',
                    'w_initializer',
                    'regularizer_const',
                    'p_noise',
                    'n_mil_sample',
                    'batch_size',
                    'learning_rate',
                    'save_freq']
    
    aux_cfgs = [
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
        'w_oh_noise_backup',
        'w_hp_noise_backup',
        'w_pp_noise_backup',
        'w_pc_noise_backup', 
        'w_cp_noise_backup',
        'bq_dataset', 
        'uuid',
        'nEpo',
        'n_timesteps', 
        'steps_per_epoch',
        'save_freq_sample', 
        'eval_freq', 
        'path_model_folder',
        'path_weights_checkpoint',              
        'path_weights_list', 
        'path_plot_folder', 
        'path_weight_folder', 
        'path_history_pickle', 
        'saved_epoch_list'
        ]
    
    all_cfgs_name = minimal_cfgs + aux_cfgs

    def __init__(self, json_file=None, bypass_chk=False, **kwargs):
        # Validate json file
        if type(json_file) == str and json_file.endswith('.json'):
            with open(json_file) as f:
                kwargs = json.load(f)

        # Set attributes
        invalid_keys = []
        for key, value in kwargs.items():
            if key in self.all_cfgs_name:
                setattr(self, key, value)
            else:
                invalid_keys.append(key)            
        
        if len(invalid_keys) > 0:
            print('These keys in cfg file is not valid: {}'.format(invalid_keys))
            
        # Check config structure is correct
        if not bypass_chk: self.chk_cfg()

        # Additional initialization for dictionary constructor 
        if json_file == None:
            self.init_from_dict()
        
    def init_from_dict(self):
        # Unique identifier
        import uuid
        self.uuid = uuid.uuid4().hex

        # Noise management
        self.w_pp_noise = self.p_noise
        self.w_pc_noise = self.p_noise
        self.w_cp_noise = self.p_noise
        self.w_oh_noise = 0.
        self.w_hp_noise = 0.

        self.w_oh_noise_backup = self.w_oh_noise
        self.w_hp_noise_backup = self.w_hp_noise
        self.w_pp_noise_backup = self.w_pp_noise
        self.w_pc_noise_backup = self.w_pc_noise
        self.w_cp_noise_backup = self.w_cp_noise

        # Additional convienient attributes
        self.n_timesteps = int(self.max_unit_time * (1 / self.tau))
        self.nEpo = int(self.n_mil_sample * 1e2)
        self.steps_per_epoch = int(10000 / self.batch_size)

        # Saving cfg
        self.save_freq_sample = self.save_freq * self.batch_size * self.steps_per_epoch  # For TF 2.1
        self.eval_freq = self.save_freq
        self.gen_paths()
        self.write_cfg()

    def __str__(self):
        return str(vars(self))

    def chk_cfg(self):
        # Check all ingested_keys fufill minimal cfg requirement
        assert all([x in vars(self) for x in self.minimal_cfgs])

        if self.use_semantic == True:
            assert type(self.sem_param_gf) == float
            assert type(self.sem_param_gi) == float
            assert type(self.sem_param_kf) == float
            assert type(self.sem_param_ki) == float
            assert type(self.sem_param_hf) == float
            assert type(self.sem_param_hi) == float

        if self.use_attractor == True:
            assert type(self.embed_attractor_cfg) == str
            assert type(self.embed_attractor_h5) == str

    def gen_paths(self):

        self.path_model_folder = 'models/' + self.code_name + '/'
        self.path_weight_folder = self.path_model_folder + 'weights/'
        self.path_plot_folder = self.path_model_folder + 'plots/'

        os.makedirs(self.path_weight_folder, exist_ok = True)
        os.makedirs(self.path_plot_folder, exist_ok = True)

        # For model checkpoint
        self.path_weights_checkpoint = self.path_weight_folder + 'ep{epoch:04d}.h5'
        self.path_history_pickle = self.path_model_folder + 'history.pickle'

        self.path_weights_list = []
        self.saved_epoch_list = []
        
        for epoch in range(1, 11):
            self.path_weights_list.append(self.path_weight_folder + 'ep' + str(epoch).zfill(4) + '.h5')
            self.saved_epoch_list.append(epoch)
            
        for epoch in range(10+self.save_freq, self.nEpo + 1, self.save_freq):
            self.path_weights_list.append(self.path_weight_folder + 'ep' + str(epoch).zfill(4) + '.h5')
            self.saved_epoch_list.append(epoch)

    def noise_off(self):
        self.w_oh_noise = 0.
        self.w_hp_noise = 0.
        self.w_pp_noise = 0.
        self.w_pc_noise = 0.
        self.w_cp_noise = 0.

    def noise_on(self):
        # Noise is on by default
        self.w_oh_noise = self.w_oh_noise_backup
        self.w_hp_noise = self.w_hp_noise_backup
        self.w_pp_noise = self.w_pp_noise_backup
        self.w_pc_noise = self.w_pc_noise_backup
        self.w_cp_noise = self.w_cp_noise_backup

    def write_cfg(self):      
        with open(self.path_model_folder + 'model_config.json', 'w') as f:
            json.dump(vars(self), f)


class connect_gbq():
    # Connect to google big query database
    def __init__(self, pid='triangle-272405'):
        """
        Store connection infos
        """
        from google.oauth2 import service_account
        self.pid = pid
        self.credentials = service_account.Credentials.from_service_account_file(
            '../common/triangle-e1fd21bb86a1.json'
        )
        
    def push_cfgs(self, db_name, cfgs):
        """
        Push a multi runs batch_cfgs object to GBQ
        """
        cfgs_df = pd.DataFrame()

        for i in range(len(cfgs)):
            cfgs_df = pd.concat(
                [cfgs_df, pd.DataFrame(cfgs[i]['params'], index=[i])]
            )
        
        pandas_gbq.to_gbq(
            cfgs_df,
            destination_table=db_name + '.cfg',
            project_id=self.pid,
            if_exists='append',
            credentials=self.credentials,
            progress_bar=False
        )

    def push_all(self, db_name, cfg, strain_i_hist, grain_i_hist, verbose=False):
        """
        Push a single run to GBQ including:
        - cfg
        - Strain item history
        - Grain item history
        """
        

        if verbose: print('Writing data to Bigquery')

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

        if verbose: print('Completed') 

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
