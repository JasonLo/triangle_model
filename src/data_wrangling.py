import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output


def gen_pkey(p_file="/home/jupyter/tf/dataset/mappingv2.txt"):
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

    assert implementation in ["log", "hs04", "jay", "chang", "experimental", "wf_linear_cutoff"]
    compressed_wf = None

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

        # Speed scaling factor (g, how fast the training set grow)
        g = 2  # For now
        progress *= g

        # Trim continuously
        clip_wf[pct > progress] = 0

        if verbose:
            print(f"minimum pct = {pct.min()}, max pct = {pct.max()}")
            print(f"Current progress: {progress}")
            print(f"Number of selected item: {sum(clip_wf > 0)}")
            print(f"Selected words: {df_train.word[clip_wf > 0]}")
            clear_output(wait=True)

        # Sqrt compression
        compressed_wf = np.sqrt(clip_wf)



    if implementation == "wf_linear_cutoff":
        """ Continuous sampling set with raw frequency as cutoff
        """
        # Top Clipping 30k
        clip_wf = df_train.wf.clip(0, 30000)

        # Monitor training progress
        progress = ingested_training_sample / max_sample

        # Speed scaling factor (g, how fast the training set grow)
        g = 2 
        progress *= g
        progress = np.clip(progress, 0, 1)

        # Scale descending clip-wf (similar to pct)
        scale_clip_wf = 1. - clip_wf/30000.

        # Trim continuously
        clip_wf[scale_clip_wf > progress] = 0

        if verbose:
            print(f"Current progress: {progress}")
            print(f"min scale_clip_wf = {scale_clip_wf.min()}")
            print(f"max scale_clip_wf = {scale_clip_wf.max()}")
            print(f"Number of selected item: {sum(clip_wf > 0)}")
            clear_output(wait=True)

        # Sqrt compression
        compressed_wf = np.sqrt(clip_wf)

    return np.array(compressed_wf/np.sum(compressed_wf), dtype="float32")


class Sampling:
    def __init__(self, cfg, data, debugging=False):
        self.cfg = cfg
        self.data = data
        self.debugging = debugging
        self.ingested_training_sample = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_stage = 0
        np.random.seed(cfg.rng_seed)

        # For debugging only
        if self.debugging:
            self.dynamic_corpus = dict.fromkeys(self.data.df_train.word, 0)
            self.debug_log_dynamic_wf = pd.DataFrame(
                index=self.data.df_train.word)  # Copy word as index
            self.debug_log_epoch = []
            self.debug_log_corpus_size = []

    def semantic_formula(self, e, t, f, i, gf, gi, kf, ki, tmax=3.8,
                         mf=4.4743, sf=2.4578, mi=4.1988, si=1.0078, hf=0, hi=0):
        # Semantic refresh V1

        numer_f = gf * e * np.log(f+2)
        denom_f = e * np.log(f+2) + kf

        return (t/tmax)*(numer_f/denom_f)

    def get_semantic_input_from_idx(self, idx):
        """ return pho x theoretical semantic_input
        """
        batch_semantic_input = np.zeros(
            (self.cfg.batch_size, self.cfg.n_timesteps, self.cfg.output_dim))

        for t in range(self.cfg.n_timesteps):
            semantic_input_at_tick_t = self.semantic_formula(
                e=self.current_epoch,
                t=t * self.cfg.tau,
                # Semantic equation is using static word frequency now
                f=self.data.wf[idx],
                i=self.data.img[idx],
                gf=self.cfg.sem_param_gf,
                gi=self.cfg.sem_param_gi,
                kf=self.cfg.sem_param_kf,
                ki=self.cfg.sem_param_ki,
                hf=self.cfg.sem_param_hf,
                hi=self.cfg.sem_param_hi,
                tmax=self.cfg.max_unit_time - self.cfg.tau
            )

            batch_semantic_input[:, t, :] = np.tile(
                np.expand_dims(semantic_input_at_tick_t, 1), [
                    1, self.cfg.output_dim]
            )

        return batch_semantic_input

    def sample_generator(self, verbose=False):
        """Dimension guide: (batch_size, timesteps, p_nodes)"""
        while True:
            # Start counting Epoch and Batch
            if self.current_batch % self.cfg.steps_per_epoch == 0:
                self.current_epoch += 1

                if self.debugging:
                    # Snapshot dynamic corpus

                    # epoch
                    self.debug_log_epoch.append(self.current_epoch - 1)

                    # init dynamic word frequency column
                    tmp_column_name = f"wf_at_epoch_{self.current_epoch}"
                    self.debug_log_dynamic_wf[tmp_column_name] = 0

                    # dynamic word frequency
                    for key, value in self.dynamic_corpus.items():
                        self.debug_log_dynamic_wf.loc[key, tmp_column_name] = value

                    # corpus size
                    tmp_corpus_size = sum(self.debug_log_dynamic_wf[tmp_column_name]>0)
                    self.debug_log_corpus_size.append(tmp_corpus_size)

            self.current_batch += 1

            # Get master sampling stage if using Chang's implementation
            if self.cfg.sample_name == "chang":
                # Need to minus batch_size, because the sample is
                self.current_stage = self.get_stage(
                    self.ingested_training_sample, normalize=True)
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

            # Debug log
            if self.debugging:
                for word_id in idx:
                    self.dynamic_corpus[self.data.df_train.word[word_id]] += 1

            # Copy y_train by the number of output ticks
            batch_y = [self.data.y_train[idx]] * self.cfg.output_ticks

            # Log ingested training sampling
            self.ingested_training_sample += self.cfg.batch_size

            if self.cfg.use_semantic:
                semantic_input = self.get_semantic_input_from_idx(idx)
                phonological_input = 2 * self.data.y_train[idx] - 1

                # Training set need to return 3 components (ort, sem, pho)
                yield ([self.data.x_train[idx], semantic_input, phonological_input], batch_y)
            else:
                yield (self.data.x_train[idx], batch_y)

    def get_stage(self, sample, normalize=False):
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

        if normalize: sample_cutoffs = np.divide(sample_cutoffs, 5.2) # Total training in ME10 = 5.2M

        return sum(sample > np.array(sample_cutoffs))


def test_set_input(
    x_test, x_test_wf, x_test_img, y_test, epoch, cfg, test_use_semantic
):
    """ Automatically restructure testset input vectors and calculate hypothetical semantic (if exist in the model)
    If model use semantic, we need to return a list of 3 inputs (x, s[time step varying], y), otherwise (x) is enough
    """

    from src.modeling import input_s

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


class MyData():
    """
    This object load all clean data from disk (both training set and testing sets)
    Also calculate sampling_p according to cfg.sample_name setting
    """

    def __init__(self):

        input_path = '/home/jupyter/tf/dataset/'

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
