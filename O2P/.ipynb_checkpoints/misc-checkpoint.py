
class model_cfg():
    
    def __init__(self, code_name, sample_name, sample_rng_seed, 
                 hidden_units, pho_units, cleanup_units, 
                 w_oh_noise, w_hp_noise, w_pp_noise, w_pc_noise, w_cp_noise,
                 act_p_noise,
                 tau, unit_time,
                 n_mil_sample, batch_size, rnn_activation,
                 w_initializer, learning_rate, save_freq):
        
        self.code_name = code_name

        # Sampling
        self.sample_name = sample_name
        self.sample_rng_seed = sample_rng_seed

        # Architechture
        self.hidden_units = hidden_units
        self.pho_units = pho_units
        self.cleanup_units = cleanup_units

        self.w_oh_noise = w_oh_noise
        self.w_hp_noise = w_hp_noise
        self.w_pp_noise = w_pp_noise
        self.w_pc_noise = w_pc_noise
        self.w_cp_noise = w_cp_noise
        self.act_p_noise = act_p_noise

        ## This is for switching between testing and training mode
        self.w_oh_noise_backup = self.w_oh_noise   
        self.w_hp_noise_backup = self.w_hp_noise
        self.w_pp_noise_backup = self.w_pp_noise
        self.w_pc_noise_backup = self.w_pc_noise
        self.w_cp_noise_backup = self.w_cp_noise
        self.act_p_noise_backup = self.act_p_noise

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

        self.gen_paths()
        self.gen_cfg_dict()
        self.write_cfg()
        self.write_cfg_to_bq()

    def gen_cfg_dict(self):
        self.cfg_dict = {'code_name': self.code_name, 
                        'sample_name': self.sample_name,
                        'sample_rng_seed': self.sample_rng_seed,
                        'hidden_units': self.hidden_units,
                        'pho_units': self.pho_units,
                        'cleanup_units': self.cleanup_units, 
                        'w_oh_noise': self.w_oh_noise,
                        'w_hp_noise': self.w_hp_noise,
                        'w_pp_noise': self.w_pp_noise,
                        'w_pc_noise': self.w_pc_noise,
                        'w_cp_noise': self.w_cp_noise,
                        'act_p_noise': self.act_p_noise,
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

    def write_cfg(self):
        import json
        json = json.dumps(self.cfg_dict)
        f = open(self.path_model_folder + 'model_config.json',"w")
        f.write(json)
        f.close()
        
    def write_cfg_to_bq(self):
        import pandas as pd
        import pandas_gbq
        
        project_id = 'idyllic-web-267716'
        bq_conn()
        cfg_pd = pd.DataFrame([self.cfg_dict])
        bq_table_cfg = 'batch_test.cfg'
        pandas_gbq.to_gbq(cfg_pd, destination_table=bq_table_cfg, project_id=project_id, if_exists='append')
        
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
        for epoch in range(self.save_freq, self.nEpo+1, self.save_freq):
            self.path_weights_list += [self.path_weight_folder + 'ep' + 
                                       str(epoch).zfill(4) + '.h5']

        self.strain_item_csv = self.path_model_folder + 'result_strain_item.csv'
        self.strain_epoch_csv = self.path_model_folder + 'result_strain_epoch.csv'
        self.grain_item_csv = self.path_model_folder + 'result_grain_item.csv'
        self.grain_epoch_csv = self.path_model_folder + 'result_grain_epoch.csv'


class my_data():
    def __init__(self, cfg):
        import numpy as np
        import pandas as pd

        self.sample_name = cfg.sample_name

        self.df_train = pd.read_csv('input/df_train.csv', index_col=0)
        self.df_strain = pd.read_csv('input/df_strain.csv', index_col=0)
        self.df_grain = pd.read_csv('input/df_grain.csv', index_col=0)

        self.x_train = np.load('input/x_train.npz')['data']
        self.x_strain = np.load('input/x_strain.npz')['data']
        self.x_grain = np.load('input/x_grain.npz')['data']
        self.y_train = np.load('input/y_train.npz')['data']
        self.y_strain = np.load('input/y_strain.npz')['data']
        self.y_large_grain = np.load('input/y_large_grain.npz')['data']
        self.y_small_grain = np.load('input/y_small_grain.npz')['data']

        from data_wrangling import gen_pkey
        self.phon_key = gen_pkey('input/mapping_v2.txt')

        print('==========Orthographic representation==========')
        print('x_train shape:', self.x_train.shape)
        print('x_strain shape:', self.x_strain.shape)
        print('x_grain shape:', self.x_grain.shape)

        print('\n==========Phonological representation==========')
        print(len(self.phon_key), ' phonemes: ', self.phon_key.keys())
        print('y_train shape:', self.y_train.shape)
        print('y_strain shape:', self.y_strain.shape)
        print('y_large_grain shape:', self.y_large_grain.shape)
        print('y_small_grain shape:', self.y_small_grain.shape)

        self.gen_sample_p()

    def gen_sample_p(self):
        from data_wrangling import wfManager
        wf = wfManager(self.df_train['wf'])

        if self.sample_name == 'hs04': self.sample_p = wf.samp_hs04()
        if self.sample_name == 'jay': self.sample_p = wf.samp_jay()
        if self.sample_name == 'log': self.sample_p = wf.samp_log()

            
def plot_variables(model):
    import matplotlib.pyplot as plt
    import numpy as np
    nv = len(model.trainable_variables)
    plt.figure(figsize=(20,20), facecolor='w')
    for i in range(nv):

        # Expand dimension for biases
        if model.trainable_variables[i].numpy().ndim == 1:
            plot_data = model.trainable_variables[i].numpy()[np.newaxis,:]
        else:
            plot_data = model.trainable_variables[i].numpy()

        plt.subplot(3, 3, i+1)
        plt.title(model.trainable_variables[i].name)
        plt.imshow(plot_data, cmap='jet', interpolation='nearest', aspect="auto")
        plt.colorbar()

        
def bq_conn():
    from google.oauth2 import service_account
    project_id = 'idyllic-web-267716'
    credentials = service_account.Credentials.from_service_account_file(
        '../tfrnn-0c7333c09ba9.json')


class load_attractor():
    def __init__(self, cfg, attractor_h5):
        self.cfg = cfg
        self.model = self.build_model()
        self.model.load_weights(attractor_h5)
        
        rnn_layer = self.model.get_layer('rnn')
        names = [weight.name for weight in rnn_layer.weights]
        weights = self.model.get_weights()

        for name, weight in zip(names, weights):
            print(name, weight.shape)
            if name.endswith('w_pp:0'): self.pretrained_w_pp = weight
            if name.endswith('w_pc:0'): self.pretrained_w_pc = weight
            if name.endswith('w_cp:0'): self.pretrained_w_cp = weight
            if name.endswith('bias_p:0'): self.pretrained_bias_p = weight
            if name.endswith('bias_c:0'): self.pretrained_bias_c = weight
        
        self.print_weights()
        
    def print_weights(self):
        from misc import plot_variables
        plot_variables(self.model)
    
    def build_model(self):
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Layer, Input
        from custom_layer import rnn_pho_task
        from tensorflow.keras.optimizers import Adam

        input_o = Input(shape=(250,), name='pho_task_input')
        rnn_model = rnn_pho_task(self.cfg, name='rnn')(input_o)
        model = Model(input_o, rnn_model)

        adam = Adam(learning_rate=self.cfg.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['BinaryAccuracy', 'mse'])

        model.summary()
        
        return model


