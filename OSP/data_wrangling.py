import pickle

import numpy as np
import pandas as pd
from IPython.display import clear_output


def gen_pkey(p_file="/home/jupyter/tf/common/patterns/mappingv2.txt"):
    """ Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict('list')
    return m_dict


def get_sampling_probability(df_train, implementation, stage=None, ingested_training_sample=None, max_sample=None, verbose=False):
    """ Return the sampling probability with different implementation
    Keyword arguments:
    df_train -- training set in pandas dataframe format, contain WSJ word frequency (wf) and Zeno frequency (gr*) 
    implementation -- method of sampling
    stage (Chang implementation only) -- which stage in Chang sampling (default None)
    ingested_training_sample (experimental implementation only) -- the ingested_training_sample
    max_sample -- maximum sample (for scaling progress)

    implementation details:
        1. log: simple log compression
        2. hs04: square root compression with bottom (1500) and top end (30000) clipping
        3. jay: square root compression with top end clipping (10000)
        4. chang: clip depending on stage, log compression
        5. experimental: continous shifting sample
    """

    assert implementation in ["log", "hs04", "jay", "chang", "experimental"]

    if implementation == "log":
        compressed_wf = np.log(1 + df_train.wf)

    if implementation == "hs04":
        root = np.sqrt(df_train.wf) / np.sqrt(30000)
        compressed_wf = root.clip(0.05, 1.0)

    if implementation == "jay":
        compressed_wf = np.sqrt(df_train.wf.clip(0, 10000))

    if implementation == "chang":
        if not (1 <= stage <= 14):
            raise ValueError("stage must be between 1-14")
        cutoffs = [1000, 100, 50, 25, 10, 8, 6, 5, 4, 3, 2, 1, 1, 0]
        wf = df_train['gr' + str(stage)]
        clip = wf.map(lambda x: x if (x > cutoffs[stage-1]) else 0)

        if verbose:
            print(f"Removed words with <= {cutoffs[stage-1]} wpm.")
            print(f"There are {np.sum(clip>0)} words in the training set")

        compressed_wf = np.log(clip + 1)

    if implementation == "experimental":
        """ Continuous sampling set
        """
        # Top Clipping 30k
        clip_wf = df_train.wf.clip(0, 30000)

        # Rank percent (Smaller = higher frequnecy)
        pct = clip_wf.rank(pct=True, ascending=False)
        
        # Monitor training progress, 0.03 for fast start (since progress = 0 has no word)
        progress = 0.03 + (ingested_training_sample/max_sample)

        # Trim continuously
        clip_wf[progress < pct] = 0

        if verbose:
            print(f"minimum pct = {pct.min()}, max pct = {pct.max()}")
            print(f"Current progress: {progress}")
            print(f"Number of selected item: {sum(clip_wf > 0)}")
            print(f"Selected words: {df_train.word[clip_wf > 0]}")
            clear_output(wait=True)

        # Sqrt compression
        compressed_wf = np.sqrt(clip_wf)

    return np.array(compressed_wf/np.sum(compressed_wf), dtype="float32")


class sampling:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.ingested_training_sample = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_stage = 0
        np.random.seed(cfg.rng_seed)

    def simple_sample_generator(self, verbose=False):
        """Dimension guide: (batch_size, timesteps, p_nodes)"""
        while True:
            # Start counting Epoch and Batch
            if self.current_batch % self.cfg.steps_per_epoch == 0:
                self.current_epoch += 1
            self.current_batch += 1

            # Get master sampling index
            if self.cfg.sample_name == "chang":
                # Need to minus batch_size, because the sample is
                self.current_stage = self.get_stage(
                    self.ingested_training_sample)
                if verbose:
                    print(
                        f"Stage: {self.current_stage}, Epoch: {self.current_epoch}, Sample: {self.ingested_training_sample}")
                    clear_output(wait=True)

            this_p = get_sampling_probability(
                df_train=self.data.df_train, 
                implementation=self.cfg.sample_name,
                stage=self.current_stage,
                ingested_training_sample=self.ingested_training_sample,
                max_sample=self.cfg.n_mil_sample * 1_000_000
            )

            idx = np.random.choice(
                range(len(this_p)), self.cfg.batch_size, p=this_p
            )

            batch_y = [self.data.y_train[idx]] * self.cfg.output_ticks

            # Log ingested training sampling
            self.ingested_training_sample += self.cfg.batch_size

            yield (self.data.x_train[idx], batch_y)

    def get_stage(self, sample):
        """ Get stage of training. See Monaghan & Ellis, 2010 """
        sample_cutoffs = [
            -1,  # because sample can be 0
            50_000,
            100_000,
            200_000,
            300_000,
            400_000,
            600_000,
            800_000,
            1_000_000,
            1_200_000,
            1_400_000,
            1_600_000,
            2_000_000,
            2_200_000,
        ]

        return sum(sample > np.array(sample_cutoffs))


# Input for training set
def sample_generator(cfg, data):
    """Generate training set sample
    Dimension guide: (batch_size, timesteps, nodes)"""

    from modeling import input_s

    np.random.seed(cfg.rng_seed)
    epoch = 0
    batch = 0

    while True:
        batch += 1

        # Get master sampling index
        idx = np.random.choice(
            range(len(data.sample_p)), cfg.batch_size, p=data.sample_p
        )

        # Preallocate
        batch_s = np.zeros((cfg.batch_size, cfg.n_timesteps, cfg.output_dim))

        # Time invarying output
        batch_y = [data.y_train[idx]] * cfg.output_ticks

        if cfg.use_semantic == True:
            for t in range(cfg.n_timesteps):
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
                )

                batch_s[:, t, :] = np.tile(
                    np.expand_dims(input_s_cell, 1), [1, cfg.output_dim]
                )

            yield (
                [data.x_train[idx], batch_s, 2 * data.y_train[idx] - 1], batch_y
            )

            # Counting epoch for ramping up input S
            if batch % cfg.steps_per_epoch == 0:
                epoch += 1
        else:
            # Non-semantic input
            yield (data.x_train[idx], batch_y)


def test_set_input(
    x_test, x_test_wf, x_test_img, y_test, epoch, cfg, test_use_semantic
):
    """ Automatically restructure testset input vectors and calculate hypothetical semantic (if exist in the model)
    If model use semantic, we need to return a list of 3 inputs (x, s[time step varying], y), otherwise (x) is enough
    """

    from modeling import input_s

    if cfg.use_semantic:
        batch_s = np.zeros(
            (len(x_test), cfg.n_timesteps, cfg.output_dim)
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
                    tmax=cfg.max_unit_time - cfg.tau  # zero-indexing
                )
                batch_s[:, t, :] = np.tile(
                    np.expand_dims(s_cell, 1), [1, cfg.output_dim]
                )

            batch_y = 2 * y_test - 1  # With negative teaching signal

        return [x_test, batch_s, batch_y]  # With negative teaching signal

    else:
        return x_test


class my_data():
    """
    This object load all clean data from disk (both training set and testing sets)
    Also calculate sampling_p according to cfg.sample_name setting
    """

    def __init__(self):

        input_path = '/home/jupyter/tf/common/input/'

        self.df_train = pd.read_csv(input_path + 'df_train.csv', index_col=0)
        self.x_train = np.load(input_path + 'x_train.npz')['data']
        self.y_train = np.load(input_path + 'y_train.npz')['data']

        self.df_strain = pd.read_csv(input_path + 'df_strain.csv', index_col=0)
        self.x_strain = np.load(input_path + 'x_strain.npz')['data']
        self.x_strain_wf = np.array(self.df_strain['wf'])
        self.x_strain_img = np.array(self.df_strain['img'])
        self.y_strain = np.load(input_path + 'y_strain.npz')['data']

        self.df_grain = pd.read_csv(input_path + 'df_grain.csv', index_col=0)
        self.x_grain = np.load(input_path + 'x_grain.npz')['data']
        self.x_grain_wf = np.array(self.df_grain['wf'])
        self.x_grain_img = np.array(self.df_grain['img'])
        self.y_large_grain = np.load(input_path + 'y_large_grain.npz')['data']
        self.y_small_grain = np.load(input_path + 'y_small_grain.npz')['data']

        self.df_taraban = pd.read_csv(
            input_path + 'df_taraban.csv', index_col=0)
        self.x_taraban = np.load(input_path + 'x_taraban.npz')['data']
        self.x_taraban_wf = np.array(self.df_taraban['wf'])
        self.x_taraban_img = np.array(self.df_taraban['img'])
        self.y_taraban = np.load(input_path + 'y_taraban.npz')['data']

        self.df_glushko = pd.read_csv(
            input_path + 'df_glushko.csv', index_col=0)
        self.x_glushko = np.load(input_path + 'x_glushko.npz')['data']
        self.x_glushko_wf = np.array(self.df_glushko['wf'])
        self.x_glushko_img = np.array(self.df_glushko['img'])
        f = open(input_path + 'y_glushko.pkl', "rb")
        self.y_glushko = pickle.load(f)
        f.close()
        f = open(input_path + 'pho_glushko.pkl', "rb")
        self.pho_glushko = pickle.load(f)
        f.close()

        self.phon_key = gen_pkey()

        # Elevate for easier access
        self.wf = np.array(self.df_train['wf'], dtype='float32')
        self.img = np.array(self.df_train['img'], dtype='float32')

        print('==========Orthographic representation==========')
        print('x_train shape:', self.x_train.shape)
        print('x_strain shape:', self.x_strain.shape)
        print('x_grain shape:', self.x_grain.shape)
        print('x_taraban shape:', self.x_taraban.shape)
        print('x_glushko shape:', self.x_glushko.shape)

        print('\n==========Phonological representation==========')
        print(len(self.phon_key), ' phonemes: ', self.phon_key.keys())
        print('y_train shape:', self.y_train.shape)
        print('y_strain shape:', self.y_strain.shape)
        print('y_large_grain shape:', self.y_large_grain.shape)
        print('y_small_grain shape:', self.y_small_grain.shape)
        print('y_taraban shape:', self.y_taraban.shape)
        print('y_glushko shape: ({}, {})'.format(
            len(self.y_glushko.items()), len(self.y_glushko['beed'][0])))
