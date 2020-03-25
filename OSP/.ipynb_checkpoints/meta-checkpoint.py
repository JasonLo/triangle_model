  
class model_cfg():
    # This function keep all global model setting
    # It will be use in almost every object downsteam, from modelling, evaluation, and visualization
    # Just keep model_cfg() light... and don't attach any heavy object here for efficiency
    
    def __init__(self, code_name=None, sample_name='hs04', sample_rng_seed=329, tf_rng_seed=123,
                 use_semantic=False,
                 sem_param_gf=4, sem_param_gi=1, 
                 sem_param_kf=100, sem_param_ki=100,
                 sem_param_hf=5, sem_param_hi=5,
                 o_input_dim=119, hidden_units=150, pho_units=250, cleanup_units=50, 
                 embed_attractor_cfg=None, embed_attractor_h5=None,
                 w_oh_noise=0., w_hp_noise=0., w_pp_noise=0., w_pc_noise=0., w_cp_noise=0.,
                 tau=0.2, unit_time=4.,
                 n_mil_sample=1., batch_size=128, rnn_activation='sigmoid',
                 w_initializer='glorot_uniform', learning_rate=0.01, save_freq=5,
                 bq_dataset='batch_test'):
        
        # Unique run id for easier tracking
        import uuid
        self.uuid = uuid.uuid4().hex
        
        self.code_name = code_name
        
        # Sampling
        self.sample_name = sample_name
        self.sample_rng_seed = sample_rng_seed
        self.tf_rng_seed = tf_rng_seed

        # Semantic input parameters
        self.use_semantic = use_semantic
        self.sem_param_gf = sem_param_gf
        self.sem_param_gi = sem_param_gi
        
        self.sem_param_kf = sem_param_kf
        self.sem_param_ki = sem_param_ki
        
        self.sem_param_hf = sem_param_hf 
        self.sem_param_hi = sem_param_hi
        
        # Architechture
        self.o_input_dim = o_input_dim
        self.hidden_units = hidden_units
        self.pho_units = pho_units
        self.cleanup_units = cleanup_units
        
        self.embed_attractor_cfg = embed_attractor_cfg
        self.embed_attractor_h5 = embed_attractor_h5

        self.w_oh_noise = w_oh_noise
        self.w_hp_noise = w_hp_noise
        self.w_pp_noise = w_pp_noise
        self.w_pc_noise = w_pc_noise
        self.w_cp_noise = w_cp_noise

        ## This is for switching between testing and training mode
        self.w_oh_noise_backup = self.w_oh_noise   
        self.w_hp_noise_backup = self.w_hp_noise
        self.w_pp_noise_backup = self.w_pp_noise
        self.w_pc_noise_backup = self.w_pc_noise
        self.w_cp_noise_backup = self.w_cp_noise

        self.tau = tau
        self.unit_time = unit_time
        self.n_timesteps = int(self.unit_time * (1/self.tau))

        # Training
        self.n_mil_sample = n_mil_sample
        self.nEpo = int(n_mil_sample * 1e2)                 
        self.batch_size = batch_size
        self.steps_per_epoch = int(10000/batch_size)
        self.rnn_activation = rnn_activation
        self.w_initializer = w_initializer
        self.learning_rate = learning_rate

        # Saving
        self.save_freq = save_freq          
        self.eval_freq = self.save_freq
        self.bq_dataset = bq_dataset

        if self.code_name is not None:
            self.gen_paths()
            self.gen_cfg_dict()
            self.write_cfg()

    def gen_cfg_dict(self):
        self.cfg_dict = {'uuid': self.uuid, 
                        'code_name': self.code_name,
                        'sample_name': self.sample_name,
                        'sample_rng_seed': self.sample_rng_seed,
                        'tf_rng_seed': self.tf_rng_seed,
                        'use_semantic': self.use_semantic,
                        'sem_param_gf': self.sem_param_gf,   
                        'sem_param_gi': self.sem_param_gi,
                        'sem_param_kf': self.sem_param_kf,
                        'sem_param_ki': self.sem_param_ki,
                        'sem_param_hf': self.sem_param_hf,
                        'sem_param_hi': self.sem_param_hi,
                        'o_input_dim': self.o_input_dim,
                        'hidden_units': self.hidden_units,
                        'pho_units': self.pho_units,
                        'cleanup_units': self.cleanup_units, 
                        'embed_attractor_cfg': self.embed_attractor_cfg,
                        'embed_attractor_h5': self.embed_attractor_h5,
                        'w_oh_noise': self.w_oh_noise,
                        'w_hp_noise': self.w_hp_noise,
                        'w_pp_noise': self.w_pp_noise,
                        'w_pc_noise': self.w_pc_noise,
                        'w_cp_noise': self.w_cp_noise,
                        'tau': self.tau,
                        'unit_time': self.unit_time,
                        'n_timesteps': self.n_timesteps,
                        'n_mil_sample': self.n_mil_sample, 
                        'nEpo': self.nEpo, 
                        'batch_size': self.batch_size, 
                        'steps_per_epoch': self.steps_per_epoch, 
                        'rnn_activation': self.rnn_activation, 
                        'w_initializer': self.w_initializer, 
                        'learning_rate': self.learning_rate}
        
    def noise_off(self):
        self.w_oh_noise = 0.
        self.w_hp_noise = 0.
        self.w_pp_noise = 0.
        self.w_pc_noise = 0.
        self.w_cp_noise = 0.
        
    def noise_on(self):
        # This is the default mode
        self.w_oh_noise = self.w_oh_noise_backup
        self.w_hp_noise = self.w_hp_noise_backup
        self.w_pp_noise = self.w_pp_noise_backup
        self.w_pc_noise = self.w_pc_noise_backup
        self.w_cp_noise = self.w_cp_noise_backup
        
    def load_cfg_json(self, file):
        import json
        with open(file) as json_file:
            self.cfg_dict = json.load(json_file)
            
            try:
                self.uuid = self.cfg_dict['uuid']
                self.code_name = self.cfg_dict['code_name']
                self.sample_name = self.cfg_dict['sample_name']
                self.sample_rng_seed = self.cfg_dict['sample_rng_seed']
                self.tf_rng_seed = self.cfg_dict['tf_rng_seed']
                self.use_semantic = self.cfg_dict['use_semantic']
                self.sem_param_gf = self.cfg_dict['sem_param_gf']
                self.sem_param_gi = self.cfg_dict['sem_param_gi']
                self.sem_param_kf = self.cfg_dict['sem_param_kf']
                self.sem_param_ki = self.cfg_dict['sem_param_ki']
                self.sem_param_hf = self.cfg_dict['sem_param_hf']
                self.sem_param_hi = self.cfg_dict['sem_param_hi']
                self.o_input_dim = self.cfg_dict['o_input_dim']
                self.hidden_units = self.cfg_dict['hidden_units']
                self.pho_units = self.cfg_dict['pho_units']
                self.cleanup_units = self.cfg_dict['cleanup_units']
                self.embed_attractor_cfg = self.cfg_dict['embed_attractor_cfg']
                self.embed_attractor_h5 = self.cfg_dict['embed_attractor_h5']
                self.w_oh_noise = self.cfg_dict['w_oh_noise']
                self.w_hp_noise = self.cfg_dict['w_hp_noise']
                self.w_pp_noise = self.cfg_dict['w_pp_noise']
                self.w_pc_noise = self.cfg_dict['w_pc_noise']
                self.w_cp_noise = self.cfg_dict['w_cp_noise']
                self.tau = self.cfg_dict['tau']
                self.unit_time = self.cfg_dict['unit_time']
                self.n_timesteps = self.cfg_dict['n_timesteps']
                self.n_mil_sample = self.cfg_dict['n_mil_sample']
                self.nEpo = self.cfg_dict['nEpo']
                self.batch_size = self.cfg_dict['batch_size']
                self.steps_per_epoch = self.cfg_dict['steps_per_epoch']
                self.rnn_activation = self.cfg_dict['rnn_activation']
                self.w_initializer = self.cfg_dict['w_initializer']
                self.learning_rate = self.cfg_dict['learning_rate']
            except:
                print('Caution: some parameter do not exist in json')
            
            self.gen_paths()
            self.gen_cfg_dict()

    def write_cfg(self):
        import json
        json = json.dumps(self.cfg_dict)
        f = open(self.path_model_folder + 'model_config.json',"w")
        f.write(json)
        f.close()
       
    def gen_paths(self):
        import os
        
        self.path_model_folder = 'models/'+ self.code_name + '/'
        self.path_log_folder = self.path_model_folder + 'log/'
        self.path_weight_folder = self.path_model_folder + 'weights/'
        self.path_plot_folder = self.path_model_folder + 'plots/'

        self.path_weights_checkpoint = self.path_weight_folder + 'ep{epoch:04d}.h5'
        self.path_history_pickle = self.path_model_folder + 'history.pickle'
               
        if not os.path.exists(self.path_model_folder): os.mkdir(self.path_model_folder) 
        if not os.path.exists(self.path_weight_folder): os.mkdir(self.path_weight_folder) 
        if not os.path.exists(self.path_log_folder): os.mkdir(self.path_log_folder) 
        if not os.path.exists(self.path_plot_folder): os.mkdir(self.path_plot_folder)  

        self.path_weights_list = []
        self.saved_epoch_list = []
        
        for epoch in range(self.save_freq, self.nEpo+1, self.save_freq):
            self.path_weights_list += [self.path_weight_folder + 'ep' + 
                                       str(epoch).zfill(4) + '.h5']
            
            self.saved_epoch_list.append(epoch)

        self.strain_item_csv = self.path_model_folder + 'result_strain_item.csv'
        self.strain_epoch_csv = self.path_model_folder + 'result_strain_epoch.csv'
        self.grain_item_csv = self.path_model_folder + 'result_grain_item.csv'
        self.grain_epoch_csv = self.path_model_folder + 'result_grain_epoch.csv'


def bq_conn(pid='idyllic-web-267716'):
    from google.oauth2 import service_account
    project_id = pid
    credentials = service_account.Credentials.from_service_account_file(
                 '../common/bq_credential.json')

    
def write_all_to_bq(cfg, strain_i_hist, grain_i_hist):
    import pandas as pd
    import pandas_gbq
    
    bq_conn()

    print('Writing data to Bigquery')
    
    # Config file
    pandas_gbq.to_gbq(pd.DataFrame([cfg.cfg_dict]), 
                      destination_table = cfg.bq_dataset + '.cfg', 
                      project_id='idyllic-web-267716', 
                      if_exists='append')
    
    # Strain eval
    pandas_gbq.to_gbq(strain_i_hist, 
                      destination_table = cfg.bq_dataset + '.strain', 
                      project_id='idyllic-web-267716', 
                      if_exists='append')
    
    # Grain eval
    pandas_gbq.to_gbq(grain_i_hist, 
                      destination_table = cfg.bq_dataset + '.grain', 
                      project_id='idyllic-web-267716', 
                      if_exists='append')
    
    print('Completed')
    

def read_bq_cfg(db_name):
    import pandas as pd
    import pandas_gbq
    
    bq_conn()
    
    sql = """
    SELECT * FROM `idyllic-web-267716.{}.cfg`
    """.format(db_name)

    return read_gbq(sql, project_id = 'idyllic-web-267716')
    
    
def yapf_all():
    # Code formatter using yapf
    # Look for .py in the entire working directory and format all scripts
    import os
    from yapf.yapflib.yapf_api import FormatFile

    for file in os.listdir():
        if file.endswith('.py'):
            FormatFile(file, in_place=True)


    
