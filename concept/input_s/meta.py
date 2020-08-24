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
        'bq_dataset',
        'batch_unique_setting_string',
    ]

    tmp_cfgs = ['w_oh_noise_backup',
                'w_hp_noise_backup',
                'w_pp_noise_backup',
                'w_pc_noise_backup',
                'w_cp_noise_backup',
                'bias_h_noise_backup',
                'bias_c_noise_backup',
                'bias_p_noise_backup',
                'path_model_folder',
                'path_weights_checkpoint',
                'path_weights_list',
                'path_plot_folder',
                'path_weight_folder',
                'path_log_folder',
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

        if (just_chk == False):
            self.store_noise()
            self.gen_paths()

            if (json_file == None):
                self.write_cfg()

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

        self.path_model_folder = 'models/' + self.code_name + '/'
        self.path_weight_folder = self.path_model_folder + 'weights/'
        self.path_plot_folder = self.path_model_folder + 'plots/'
        self.path_log_folder = self.path_model_folder + 'logs/'

        os.makedirs(self.path_weight_folder, exist_ok=True)
        os.makedirs(self.path_plot_folder, exist_ok=True)
        os.makedirs(self.path_log_folder, exist_ok=True)

        # For model checkpoint
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

    def write_cfg(self):

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

