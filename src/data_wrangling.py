# This script contain a set of custom functions for managing representations

import pickle, gzip, os
import numpy as np
import pandas as pd
from IPython.display import clear_output


def gen_pkey(p_file="/home/jupyter/tf/dataset/mappingv2.txt"):
    """Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict("list")
    return m_dict


class Sampling:
    """Full function sampling class, can be quite slow, but contain a dynamic logging function,
    mainly for model v3.x (using equation to manage semantic input)
    """

    def __init__(self, cfg, data, debugging=False):
        self.cfg = cfg
        self.data = data
        self.debugging = debugging
        self.ingested_training_sample = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_stage = 0
        self.dynamic_corpus = dict.fromkeys(self.data.df_train.word, 0)
        np.random.seed(cfg.rng_seed)

        # For debugging only
        if self.debugging:
            self.debug_wf = []
            self.debug_sem = []
            self.debug_epoch = []
            self.debug_corpus_size = []

    def set_semantic_parameters(self, **kwargs):
        self.semantic_params = kwargs

    def semantic_input(self, f):
        """Semantic equation"""
        g = self.semantic_params["g"]
        k = self.semantic_params["k"]
        # h = self.semantic_params["h"]
        w = self.semantic_params["w"]

        # numer = g * np.log(10**w * f + h)
        # denom = np.log(10**w * f + h) + k
        numer = g * (f * w)
        denom = (w * f) + k
        return numer / denom

    def sample_generator(self, x, y, dryrun=False):
        """Generator for training data
        x: input str ("ort" / "pho" / "sem")
        y: output str ("ort" / "pho" / "sem"), can be a list
        dryrun: only sample the words without outputing representations
        representation dimension guide: (batch_size, timesteps, output_nodes)
        """
        while True:
            # Start counting Epoch and Batch
            if self.current_batch % self.cfg.steps_per_epoch == 0:

                if self.debugging:
                    # Snapshot dynamic corpus
                    self.debug_epoch.append(self.current_epoch)
                    self.debug_wf.append(self.dynamic_corpus.copy())
                    self.debug_corpus_size.append(
                        sum(wf > 0 for wf in self.debug_wf[-1].values())
                    )
                    # self.debug_sem.append(
                    #     {
                    #         k: self.semantic_input(v)
                    #         for k, v in self.debug_wf[-1].items()
                    #     }
                    # )

                self.current_epoch += 1

            self.current_batch += 1

            # Get master sampling stage if using Chang's implementation
            if self.cfg.sample_name == "chang_jml":
                # Need to minus batch_size, because the sample is
                self.current_stage = self.get_stage(
                    self.ingested_training_sample, normalize=False
                )

                this_p = self.get_sampling_probability(
                    df_train=self.data.df_train,
                    implementation=self.cfg.sample_name,
                    stage=self.current_stage,
                    ingested_training_sample=self.ingested_training_sample,
                )

            elif self.cfg.sample_name == "flexi_rank":
                this_p = self.get_sampling_probability(
                    df_train=self.data.df_train,
                    implementation=self.cfg.sample_name,
                    wf_low_clip=self.cfg.wf_low_clip,
                    wf_high_clip=self.cfg.wf_high_clip,
                    compression=self.cfg.wf_compression,
                    sampling_plateau=self.cfg.sampling_plateau,
                    ingested_training_sample=self.ingested_training_sample,
                )

            else:
                this_p = self.get_sampling_probability(
                    df_train=self.data.df_train, implementation=self.cfg.sample_name
                )

            # Sample
            idx = np.random.choice(range(len(this_p)), self.cfg.batch_size, p=this_p)
            words = self.data.df_train.word.loc[idx]

            # Update dynamic corpus
            for word in words:
                self.dynamic_corpus[word] += 1

            # Log ingested training sampling
            self.ingested_training_sample += self.cfg.batch_size

            if dryrun:
                yield (self.current_batch)
            else:
                # Real output
                batch_x = self.data.np_representations[x][idx]

                if type(y) is list:
                    batch_y = [
                        [self.data.np_representations[yi][idx]] * self.cfg.inject_error_ticks
                        for yi in y
                    ]
                else:
                    batch_y = [
                        self.data.np_representations[y][idx]
                    ] * self.cfg.inject_error_ticks
                yield (batch_x, batch_y)

    def get_stage(self, sample, normalize=False):
        """ Get stage of training. See Monaghan & Ellis, 2010 """
        sample_cutoffs = [
            -1,  # because sample can be 0
            100_000,
            150_000,
            210_000,
            260_000,
            300_000,
            380_000,
            460_000,
            540_000,
            620_000,
            700_000,
            780_000,
            860_000,
            940_000,
        ]

        if normalize:
            # Total training in ME10 = 5.2M
            sample_cutoffs = np.divide(sample_cutoffs, 5.2)

        return sum(sample > np.array(sample_cutoffs))

    @staticmethod
    def get_sampling_probability(
        df_train,
        implementation,
        wf_low_clip=None,
        wf_high_clip=None,
        compression=None,
        sampling_plateau=None,
        stage=None,
        ingested_training_sample=None,
        vocab_size=None,
        verbose=True,
    ):
        """Return the sampling probability with different implementation
        Keyword arguments:
        df_train -- training set in pandas dataframe format, contain WSJ word frequency (wf) and Zeno frequency (gr*)
        implementation -- method of sampling
        wf_low_clip -- bottom freqeuncy clipping
        wf_high_clip -- upper frequency clipping
        compression -- log or square root frequency compression
        sampling_plateau -- at what number of sample, the corpus will fully open
        stage (Chang implementation only) -- which stage in Chang sampling (default None)
        ingested_training_sample (experimental implementation only) -- the ingested_training_sample

        implementation details:
            1. log: simple log compression
            2. hs04: square root compression with bottom (1500) and top end (30000) clipping
            3. jay: square root compression with top end clipping (10000)
            4. chang_jml: clip depending on stage, log compression
            5. flexi_rank: a modifyied smooth rank-based sampler 
            6. wf_linear_cutoff: continous shifting sample by raw word frequency
        """
        
        assert implementation in [
            "log",
            "hs04",
            "jay",
            "chang_jml",
            "chang_ssr",
            "flexi_rank"
        ]
        compressed_wf = None

        if implementation == "log":
            compressed_wf = np.log(1 + df_train.wf)

        if implementation == "hs04":
            root = np.sqrt(df_train.wf) / np.sqrt(30000)
            compressed_wf = root.clip(0.05, 1.0)

        if implementation == "jay":
            compressed_wf = np.sqrt(df_train.wf.clip(0, 10000))

        if implementation == "chang_jml":
            if not (1 <= stage <= 14):
                raise ValueError("stage must be between 1-14")
            cutoffs = [1000, 100, 50, 25, 10, 8, 6, 5, 4, 3, 2, 1, 1, 0]
            wf = df_train["gr" + str(stage)]
            clip = wf.map(lambda x: x if (x > cutoffs[stage - 1]) else 0)

            if verbose:
                print(f"Removed words with <= {cutoffs[stage-1]} wpm.")
                print(f"There are {np.sum(clip>0)} words in the training set")

            compressed_wf = np.log(clip + 1)

        if implementation == "chang_ssr":
            wf = df_train.wf.copy()
            root = np.sqrt(wf) / np.sqrt(30000)
            compressed_wf = root.clip(0.0, 1.0)

            sel = df_train.wf.rank(ascending=False) <= vocab_size
            compressed_wf[~sel] = 0

        if implementation == "flexi_rank":
            """Flexible rank sampler"""
            clip_wf = df_train.wf.clip(wf_low_clip, wf_high_clip)
            pct = clip_wf.rank(pct=True, ascending=False)
            progress = 0.03 + (ingested_training_sample / sampling_plateau)
            clip_wf[pct > progress] = 0
            if compression == "log":
                compressed_wf=np.log(clip_wf+1)
                # Must use +1 here, otherwise clip wf = 0 still has chance to get sampled
            elif compression == "root":
                compressed_wf = np.sqrt(clip_wf)


        return np.array(compressed_wf / np.sum(compressed_wf), dtype="float32")

class FastSampling_uniform:
    """Performance oriented sample generator
    A simplified version of Sampling() with uniform p
    """

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        np.random.seed(cfg.rng_seed)

    def sample_generator(self, x, y, x_ticks=None, y_ticks=None):
        """Generator for training data
        x: input str ("ort" / "pho" / "sem")
        y: output str ("ort" / "pho" / "sem") can be a list
        representation dimension guide: (batch_size, timesteps, output_nodes)
        """

        if x_ticks is None:
            x_ticks = self.cfg.n_timesteps

        if y_ticks is None:
            y_ticks = self.cfg.inject_error_ticks

        while True:

            # Sample
            idx = np.random.choice(
                len(self.data.np_representations[x]), self.cfg.batch_size
            )
            batch_x = [self.data.np_representations[x][idx]] * x_ticks

            if type(y) is list:
                # Multi output as a list
                batch_y = [
                    [self.data.np_representations[yi][idx]] * y_ticks for yi in y
                ]
            else:
                # Single output
                batch_y = [self.data.np_representations[y][idx]] * y_ticks

            yield (batch_x, batch_y)

                       
class FastSampling:
    """Performance oriented sample generator
    A simplified version of Sampling() for hs04 model
    """

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        np.random.seed(cfg.rng_seed)



    def sample_generator(self, x, y, x_ticks=None, y_ticks=None, training_set=None, implementation=None):
        """Generator for training data
        x: input str ("ort" / "pho" / "sem")
        y: output str ("ort" / "pho" / "sem") can be a list
        representation dimension guide: (batch_size, timesteps, output_nodes)
        """

        if x_ticks is None:
            x_ticks = self.cfg.n_timesteps

        if y_ticks is None:
            y_ticks = self.cfg.inject_error_ticks
            
        if training_set is None:
            training_set = self.data.df_train
            
        if implementation is None:
            implementation = self.cfg.sample_name

        while True:

            # Get master sampling stage if using Chang's implementation
            if implementation == "chang_jml":
                # Need to minus batch_size, because the sample is
                self.current_stage = self.get_stage(
                    self.ingested_training_sample, normalize=False
                )

                this_p = Sampling.get_sampling_probability(
                    df_train=training_set,
                    implementation=implementation,
                    stage=self.current_stage,
                    ingested_training_sample=self.ingested_training_sample,
                )

            elif implementation == "flexi_rank":
                this_p = Sampling.get_sampling_probability(
                    df_train=training_set,
                    implementation=implementation,
                    sampling_plateau=self.cfg.sampling_plateau,
                    ingested_training_sample=self.ingested_training_sample,
                )

            elif implementation in ("log", "hs04", "jay"):
                this_p = Sampling.get_sampling_probability(
                    df_train=training_set, 
                    implementation=implementation
                )

            elif implementation == "chang_ssr":
                this_p = Sampling.get_sampling_probability(
                    df_train=training_set,
                    implementation=implementation,
                    vocab_size=self.cfg.oral_vocab_size,
                )

            else:
                this_p = None
                


            # Sample
            idx = np.random.choice(range(len(this_p)), self.cfg.batch_size, p=this_p)
            batch_x = [self.data.np_representations[x][idx]] * x_ticks

            if type(y) is list:
                # Multi output as a list
                batch_y = [
                    [self.data.np_representations[yi][idx]] * y_ticks for yi in y
                ]
            else:
                # Single output
                batch_y = [self.data.np_representations[y][idx]] * y_ticks

            yield (batch_x, batch_y)


class MyData:
    """
    This object load all clean data from disk (both training set and testing sets)
    Also calculate sampling_p according to cfg.sample_name setting
    """

    def __init__(self):

        self.input_path = "/home/jupyter/tf/dataset/"

        # init an empty testset dict for new testset format
        # first level: testset name
        # second level (in each testset): word, ort, pho, sem
        self.testsets = {}
        self.load_testsets()

        self.df_train = pd.read_csv(
            os.path.join(self.input_path, "df_train.csv"), index_col=0
        )
        self.ort_train = np.load(os.path.join(self.input_path, "ort_train.npz"))["data"]
        self.pho_train = np.load(os.path.join(self.input_path, "pho_train.npz"))["data"]
        self.sem_train = np.load(os.path.join(self.input_path, "sem_train.npz"))["data"]

        self.np_representations = {
            "ort": self.ort_train,
            "pho": self.pho_train,
            "sem": self.sem_train,
        }

        self.df_strain = pd.read_csv(
            os.path.join(self.input_path, "df_strain.csv"), index_col=0
        )
        # self.ort_strain = np.load(os.path.join(self.input_path, "ort_strain.npz"))[
        #     "data"
        # ]
        # self.ort_strain_wf = np.array(self.df_strain["wf"])
        # self.ort_strain_img = np.array(self.df_strain["img"])
        # self.pho_strain = np.load(os.path.join(self.input_path, "pho_strain.npz"))[
        #     "data"
        # ]
        # self.sem_strain = np.load(os.path.join(self.input_path, "sem_strain.npz"))[
        #     "data"
        # ]

        self.df_grain = pd.read_csv(
            os.path.join(self.input_path, "df_grain.csv"), index_col=0
        )
        # self.ort_grain = np.load(os.path.join(self.input_path, "ort_grain.npz"))["data"]
        # self.ort_grain_wf = np.array(self.df_grain["wf"])
        # self.ort_grain_img = np.array(self.df_grain["img"])
        # self.pho_large_grain = np.load(
        #     os.path.join(self.input_path, "pho_large_grain.npz")
        # )["data"]
        # self.pho_small_grain = np.load(
        #     os.path.join(self.input_path, "pho_small_grain.npz")
        # )["data"]


        self.df_taraban = pd.read_csv(
            os.path.join(self.input_path, "df_taraban.csv"), index_col=0
        )
        # self.ort_taraban = np.load(os.path.join(self.input_path, "ort_taraban.npz"))[
        #     "data"
        # ]
        # self.ort_taraban_wf = np.array(self.df_taraban["wf"])
        # self.ort_taraban_img = np.array(self.df_taraban["img"])
        # self.pho_taraban = np.load(os.path.join(self.input_path, "pho_taraban.npz"))[
        #     "data"
        # ]

        self.df_glushko = pd.read_csv(
            os.path.join(self.input_path, "df_glushko.csv"), index_col=0
        )
        self.x_glushko = np.load(os.path.join(self.input_path, "ort_glushko.npz"))[
            "data"
        ]
        self.x_glushko_wf = np.array(self.df_glushko["wf"])
        self.x_glushko_img = np.array(self.df_glushko["img"])

        with open(self.input_path + 'y_glushko.pkl', "rb") as f:
            self.y_glushko = pickle.load(f)

        with open(os.path.join(self.input_path, "pho_glushko.pkl"), "rb") as f:
            self.pho_glushko = pickle.load(f)


        self.phon_key = gen_pkey()

        # Elevate for easier access
        # self.wf = np.array(self.df_train["wf"], dtype="float32")
        # self.img = np.array(self.df_train["img"], dtype="float32")

        # print("==========Orthographic representation==========")
        # print("ort_train shape:", self.ort_train.shape)
        # print("ort_strain shape:", self.ort_strain.shape)
        # print("ort_grain shape:", self.ort_grain.shape)
        # print("ort_taraban shape:", self.ort_taraban.shape)
        # print("ort_glushko shape:", self.ort_glushko.shape)

        # print("\n==========Phonological representation==========")
        # print(len(self.phon_key), " phonemes: ", self.phon_key.keys())
        # print("pho_train shape:", self.pho_train.shape)
        # print("pho_strain shape:", self.pho_strain.shape)
        # print("pho_large_grain shape:", self.pho_large_grain.shape)
        # print("pho_small_grain shape:", self.pho_small_grain.shape)
        # print("pho_taraban shape:", self.pho_taraban.shape)
        # print(
        #     "pho_glushko shape: ({}, {})".format(
        #         len(self.pho_glushko.items()), len(self.pho_glushko["beed"][0])
        #     )
        # )

        # print("\n==========Semantic representation==========")
        # print("sem_train shape:", self.sem_train.shape)
        # print("sem_strain shape:", self.sem_strain.shape)

    def create_testset_from_train_idx(self, idx):
        """Return a test set representation dictionary with word, ort, pho, sem"""
        item = list(self.df_train.loc[idx, "word"].astype("str"))
        ort = self.ort_train[idx,]
        pho = self.pho_train[idx,]
        sem = self.sem_train[idx,]
        return {"item": item, "ort": ort, "pho": pho, "sem": sem}

    def load_testsets(self):

        all_testsets = (
            "homophone",
            "non_homophone",
            "train",
            "strain_hf_con_hi",
            "strain_hf_inc_hi",
            "strain_hf_con_li",
            "strain_hf_inc_li",
            "strain_lf_con_hi",
            "strain_lf_inc_hi",
            "strain_lf_con_li",
            "strain_lf_inc_li",
            "grain_unambiguous",
            "grain_ambiguous",
            "cortese_hi_img", 
            "cortese_low_img",
            "cortese_3gp_high_img",
            "cortese_3gp_med_img",
            "cortese_3gp_low_img",
            "taraban_hf-exc",
            "taraban_hf-reg-inc",
            "taraban_lf-exc",
            "taraban_lf-reg-inc",
            "taraban_ctrl-hf-exc",
            "taraban_ctrl-hf-reg-inc",
            "taraban_ctrl-lf-exc",
            "taraban_ctrl-lf-reg-inc",
        )

        for testset in all_testsets:
            with gzip.open(
                os.path.join(self.input_path, "testsets", testset + ".pkl.gz"), "rb"
            ) as f:
                self.testsets[testset] = pickle.load(f)
