import numpy as np
import pandas as pd


def gen_pkey(p_file="../common/patterns/mappingv2.txt"):
    """
    Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict('list')
    return m_dict


class wf_manager():
    # Note: the probability must sum to 1 when passing it to np.random.choice()
    def __init__(self, wf):

        self.wf = np.array(wf)

    def wf(self):
        return self.wf

    def to_p(self, x):
        return x / np.sum(x)

    def samp_termf(self):
        return self.to_p(self.wf)

    def samp_log(self):
        log = np.log(1 + self.wf)
        return self.to_p(log)

    def samp_hs04(self):
        root = np.sqrt(self.wf) / np.sqrt(30000)
        clip = root.clip(0.05, 1.0)
        return self.to_p(clip)

    def samp_jay(self):
        cap = self.wf.clip(0, 10000)
        root = np.sqrt(cap)
        return self.to_p(root)


# Input for training set
def sample_generator(cfg, data):
    # Dimension guide: (batch_size, timesteps, nodes)
    from modeling import input_s

    np.random.seed(cfg.sample_rng_seed)
    epoch = 0
    batch = 0

    while True:
        batch += 1
        idx = np.random.choice(
            range(len(data.sample_p)), cfg.batch_size, p=data.sample_p
        )

        # Preallocate for easier indexing: (batch_size, time_step, input_dim)
        batch_s = np.zeros((cfg.batch_size, cfg.n_timesteps, cfg.pho_units))
        batch_y = []

        for t in range(cfg.n_timesteps):

            if cfg.use_semantic == True:
                input_s_cell = input_s(
                    e=epoch,
                    t=t * cfg.tau,
                    f=data.wf[idx],
                    i=data.img[idx],
                    gf=cfg.sem_param_gf,
                    gi=cfg.sem_param_gi,
                    kf=cfg.sem_param_kf,
                    ki=cfg.sem_param_ki,
                    hf=cfg.sem_param_hf,
                    hi=cfg.sem_param_hi,
                    tmax=cfg.max_unit_time - cfg.tau
                )  # Because output use zero indexing... we have to -1 step

                batch_s[:, t, :] = np.tile(
                    np.expand_dims(input_s_cell, 1), [1, cfg.pho_units]
                )

            batch_y.append(data.y_train[idx])

        if batch % cfg.steps_per_epoch == 0:
            epoch += 1  # Counting epoch for ramping up input S

        if cfg.use_semantic == True:
            yield (
                [data.x_train[idx], batch_s, 2 * data.y_train[idx] - 1], batch_y
            )  # With negative input signal
        else:
            yield (data.x_train[idx], batch_y)


def test_set_input(
    x_test, x_test_wf, x_test_img, y_test, epoch, cfg, test_use_semantic
):
    # We need to separate whether the model use semantic by cfg.use_semantic
    # And whether the test set has semantic input by test_use_semantic
    # If model use semantic, we need to return a list of 3 inputs (x, s[time step varying], y), otherwise (x) is enough
    from modeling import input_s

    if cfg.use_semantic:
        batch_s = np.zeros(
            (len(x_test), cfg.n_timesteps, cfg.pho_units)
        )  # Fill batch_s with Plaut S formula
        batch_y = np.zeros_like(y_test)

        if test_use_semantic:
            for t in range(cfg.n_timesteps):
                s_cell = input_s(
                    e=epoch,
                    t=t * cfg.tau,
                    f=x_test_wf,
                    i=x_test_img,
                    gf=cfg.sem_param_gf,
                    gi=cfg.sem_param_gi,
                    kf=cfg.sem_param_kf,
                    ki=cfg.sem_param_ki,
                    hf=cfg.sem_param_hf,
                    hi=cfg.sem_param_hi,
                    tmax=cfg.max_unit_time - cfg.tau
                )  # Because of zero indexing
                batch_s[:, t, :] = np.tile(
                    np.expand_dims(s_cell, 1), [1, cfg.pho_units]
                )

            batch_y = 2 * y_test - 1  # With negative teaching signal

        return [x_test, batch_s, batch_y]  # With negative teaching signal

    else:
        return x_test


class my_data():
    def __init__(self, cfg):

        self.sample_name = cfg.sample_name
        self.x_name = cfg.x_name
        self.y_name = cfg.y_name

        input_path = '../common/input/'

        self.df_train = pd.read_csv('df_train_4721.csv', index_col=0)
        self.x_train = np.load(input_path + self.x_name + '.npz')['data']
        self.y_train = np.load(input_path + self.y_name + '.npz')['data']

        self.df_strain = pd.read_csv(input_path + 'df_strain.csv', index_col=0)
        self.df_grain = pd.read_csv(input_path + 'df_grain.csv', index_col=0)

        self.x_strain = np.load(input_path + 'x_strain.npz')['data']
        self.x_strain_wf = np.array(self.df_strain['wf'])
        self.x_strain_img = np.array(self.df_strain['img'])
        self.y_strain = np.load(input_path + 'y_strain_tasa.npz')['data']


        from data_wrangling import gen_pkey
        self.phon_key = gen_pkey()

        print('==========Orthographic representation==========')
        print('x_train shape:', self.x_train.shape)
        print('x_strain shape:', self.x_strain.shape)

        print('\n==========Phonological representation==========')
        print(len(self.phon_key), ' phonemes: ', self.phon_key.keys())
        print('y_train shape:', self.y_train.shape)
        print('y_strain shape:', self.y_strain.shape)
        
        # Select WF
        ratio = self.df_train['lwf'] / np.log(30000)
        clip = ratio.clip(0.05, 1.0)
        self.sample_p =  clip / np.sum(clip)
