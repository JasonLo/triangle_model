# This script contain a set of custom functions for managing representations

import pickle, gzip, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import modeling
import helper as H


def gen_pkey(p_file="/home/jupyter/tf/dataset/mappingv2.txt"):
    """Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict("list")
    return m_dict

class Sampler:
    """v2 Sampler"""

    def __init__(self, cfg, data):

        # Get necessary environment config from cfg object
        self.tasks = cfg.tasks
        self.wf_clip_low = cfg.wf_clip_low
        self.wf_clip_high = cfg.wf_clip_high
        self.wf_compression = cfg.wf_compression
        self.oral_start_pct = cfg.oral_start_pct
        self.oral_end_pct = cfg.oral_end_pct 
        self.oral_sample = cfg.oral_sample
        self.oral_tasks_ps = cfg.oral_tasks_ps
        self.transition_sample = cfg.transition_sample
        self.reading_sample = cfg.reading_sample
        self.reading_tasks_ps = cfg.reading_tasks_ps
        self.batch_size = cfg.batch_size
        self.n_timesteps = cfg.n_timesteps
        self.inject_error_ticks = cfg.inject_error_ticks
        
        np.random.seed(cfg.rng_seed)
        
        self.data = data
        self.current_sample = 0

        # Basic convienient variables
        self._calculate_aux_variables()

        # Task probability dictionary (batch, ps)
        self.ps = {}
        self._calculate_task_ps()

        # Progress dictionary 
        self.progress = {}
        self._calculate_progress_dict()

    def plot(self):
        """Plot an easy to understand environment progression figure"""
        plt.plot(self.progress['oral'], label='oral corpus')
        plt.plot(self.progress['reading'], label='reading corpus')

        reading_p = [self.task_ps[i][-1] if type(self.task_ps[i]) is tuple else self.task_ps[i] for i in range(self.total_batches)]
        plt.plot(reading_p, label='reading_p', linestyle='dashdot', color='black')

        plt.axvline(x=self.oral_batches, ymin=0, ymax=1, linestyle = 'dotted', color='red', label='transition start')
        plt.axvline(x=self.oral_batches + self.transition_batches, ymin=0, ymax=1, linestyle = 'dotted', color='green', label = 'transition end')
        plt.text(x=10, y=0.8, s=f"oral phase task ps \n{self.oral_tasks_ps}")
        plt.text(x=self.total_batches*0.5, y=0.8, s=f"reading phase task ps \n(after transition) \n{self.reading_tasks_ps}")
        plt.xlabel('batch')

        plt.legend()
        plt.title('Corpus opening progression (%)')
        plt.show()

    def _calculate_aux_variables(self):
        self.total_sample = self.oral_sample + self.reading_sample

        self.total_batches = self.sample_to_batch(self.total_sample)
        self.oral_batches = self.sample_to_batch(self.oral_sample)
        self.transition_batches = self.sample_to_batch(self.transition_sample)
        self.remaining_batches = self.total_batches - self.oral_batches - self.transition_batches

        # Word frequency related
        if self.wf_clip_low is None:
            self.wf_clip_low = 0
        
        if self.wf_clip_high is None:
            self.wf_clip_high = 999999999            
            
        clip_wf = self.data.df_train.wf.clip(self.wf_clip_low, self.wf_clip_high).copy()
        self.rank_pct_wf = clip_wf.rank(pct=True, ascending=False)

        assert self.wf_compression in ('log', 'root')
        self.compressed_wf = np.log(clip_wf + 1) if self.wf_compression == 'log' else np.sqrt(clip_wf)

    def _calculate_task_ps(self):
        """Task probabililty store all task probabilities in a single dictionary (epoch, ps) """

        # Oral
        self.task_ps = {i:self.oral_tasks_ps for i in range(self.oral_batches)}

        # Transition
        tps = np.linspace(self.oral_tasks_ps, self.reading_tasks_ps, self.transition_batches)
        tran_ps = {i + self.oral_batches: tuple(tps[i]) for i in range(self.transition_batches)}
        self.task_ps.update(tran_ps)

        # Remaining
        remain_ps = {i + self.oral_batches + self.transition_batches: self.reading_tasks_ps for i in range(self.remaining_batches)}
        self.task_ps.update(remain_ps)

    def _calculate_progress_dict(self):
        """Progress dictionary storing all %max progress at each epoch in oral and reading stage"""
        # Oral progress
        opg = np.linspace(self.oral_start_pct, self.oral_end_pct, self.oral_batches)
        self.progress['oral'] = self._extrapolate_ps(opg)

        # Reading progress
        self.progress['reading'] = self._right_shift_ps(self.progress['oral'])         

    def _extrapolate_ps(self, array):
        """Extrapolate an array to the number of batches long"""
        d = array[1] - array[0]
        e = array[-1]
        n = len(array)
        ex_array = [array[i] if i < n  else e + (i-n) * d for i in range(self.total_batches)]
        return np.clip(ex_array, 0, 1)

    def _right_shift_ps(self, oral_progress):
        """Convert oral ps to reading ps
        Since reading lag behide oral task by a constant, we just need to shift oral progress to the right
        to get reading progress
        """
        n = self.oral_batches
        beginning = np.zeros(n)
        return np.concatenate([beginning, oral_progress[:-n]])

    def wf_to_ps(self, wf):
        """convert squashed compressed word frequncy to probabilty"""
        return np.array(wf/np.sum(wf), dtype="float32")

    def sample_to_batch(self, sample):
        """Convert sample to batch in 0-indexing format"""
        return int(sample/self.batch_size)

    def get_sampling_p(self, current_sample, task):
        current_batch = self.sample_to_batch(current_sample)

        if task == 'triangle':
            progress = self.progress['reading'][current_batch]
        else:
            progress = self.progress['oral'][current_batch]

        # Create selection mask
        mask = self.rank_pct_wf < progress
    
        return self.wf_to_ps(self.compressed_wf * mask)

    def generator(self):
        """ Generator that draw task and sample idx 
        """
        x_ticks = self.n_timesteps
        y_ticks = self.inject_error_ticks
        
        while True:
            current_batch = self.sample_to_batch(self.current_sample)
            task = np.random.choice(self.tasks, p=self.task_ps[current_batch]) if type(self.tasks) is tuple else self.tasks
            idx = np.random.choice(self.data.df_train.index, self.batch_size, p=self.get_sampling_p(self.current_sample, task))
            
            x, y = modeling.IN_OUT[task]
            batch_x = [self.data.np_representations[x][idx]] * x_ticks

            if type(y) is list:
                    batch_y = {yi: [self.data.np_representations[yi][idx]] * y_ticks for yi in y}
            else:
                # Single output
                batch_y = [self.data.np_representations[y][idx]] * y_ticks
            
            
            self.current_sample += self.batch_size
            yield task, idx, batch_x, batch_y
     


class OldSampling:
    """Full function sampling class, can be quite slow, but contain a dynamic logging function,
    mainly for model v3.x (using equation to manage semantic input)
    Kind of slow... and many experimental features
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
                    wf_compression=self.cfg.wf_compression,
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
                        [self.data.np_representations[yi][idx]]
                        * self.cfg.inject_error_ticks
                        for yi in y
                    ]
                else:
                    batch_y = [
                        self.data.np_representations[y][idx]
                    ] * self.cfg.inject_error_ticks
                yield (batch_x, batch_y)

    def get_stage(self, sample, normalize=False):
        """Get stage of training. See Monaghan & Ellis, 2010"""
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
        wf_compression=None,
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
        wf_compression -- log or square root frequency compression
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
            "flexi_rank",
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
            clip_wf = df_train.wf.clip(wf_low_clip, wf_high_clip).copy()
            pct = clip_wf.rank(pct=True, ascending=False)
            progress = 0.02 + (ingested_training_sample / sampling_plateau)
            clip_wf[pct > progress] = 0
            if wf_compression == "log":
                compressed_wf = np.log(clip_wf + 1)
                # Must use +1 here, otherwise clip wf = 0 still has chance to get sampled
            elif wf_compression == "root":
                compressed_wf = np.sqrt(clip_wf)

        return np.array(compressed_wf / np.sum(compressed_wf), dtype="float32")


class FastSampling:
    """Performance oriented sample generator
    A simplified version of Sampling() for hs04 model
    """

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.ingested_training_sample = 0
        self.current_sample = 0
        self.current_batch = 0
        self.current_epoch = 0
        self.dynamic_corpus = dict.fromkeys(self.data.df_train.word, 0)
        self.corpus_snapshots = pd.DataFrame(index=self.data.df_train.word)
        np.random.seed(cfg.rng_seed)

    def _update_metadata(self, idx):
        """Update dynamic corpus, corpus_snapshot, injested training sample with actual exposure"""
        exposed_words = self.data.df_train.word.loc[idx]

        for word in exposed_words:
            self.dynamic_corpus[word] += 1

        self.ingested_training_sample += len(idx)
        self.current_batch += 1

        if self.current_batch % self.cfg.steps_per_epoch == 0:
            # At the end of each epoch snapshot dynamic corpus
            self.current_epoch += 1
            self.corpus_snapshots[f"epoch_{self.current_epoch:04d}"] = [
                self.dynamic_corpus[k] for k in self.corpus_snapshots.index
            ]

    def sample_generator(self, x, y, training_set=None, implementation=None):
        """Generator for training data
        x: input str ("ort" / "pho" / "sem")
        y: output str ("ort" / "pho" / "sem") can be a list
        representation dimension guide: (batch_size, timesteps, output_nodes)
        """

        if training_set is None:
            training_set = self.data.df_train

        if implementation is None:
            implementation = self.cfg.sample_name

        # fail safe
        assert implementation in ("flexi_rank", "log", "hs04", "jay")

        while True:

            this_p = OldSampling.get_sampling_probability(
                df_train=training_set,
                implementation=implementation,
                wf_low_clip=self.cfg.wf_low_clip,
                wf_high_clip=self.cfg.wf_high_clip,
                wf_compression=self.cfg.wf_compression,
                sampling_plateau=self.cfg.sampling_plateau,
                ingested_training_sample=self.ingested_training_sample,
            )

            # Sample
            idx = np.random.choice(training_set.index, self.cfg.batch_size, p=this_p)
            batch_x = [self.data.np_representations[x][idx]] * self.cfg.n_timesteps

            if type(y) is list:
                # Multi output as a list
                batch_y = [
                    [self.data.np_representations[yi][idx]]
                    * self.cfg.inject_error_ticks
                    for yi in y
                ]
            else:
                # Single output
                batch_y = [
                    self.data.np_representations[y][idx]
                ] * self.cfg.inject_error_ticks

            self._update_metadata(idx)

            yield (batch_x, batch_y)


class MyData:
    """
    This object load all clean data from disk (both training set and testing sets)
    Also calculate sampling_p according to cfg.sample_name setting
    """

    def __init__(self, input_path="/home/jupyter/tf/dataset/"):

        self.input_path = input_path

        # init an empty testset dict for new testset format
        # first level: testset name
        # second level (in each testset): word, ort, pho, sem
        self.testsets = {}
        self.load_all_testsets()

        self.np_representations = {
            "ort": np.load(os.path.join(self.input_path, "ort_train.npz"))["data"],
            "pho": np.load(os.path.join(self.input_path, "pho_train.npz"))["data"],
            "sem": np.load(os.path.join(self.input_path, "sem_train.npz"))["data"],
        }

        self.df_train = pd.read_csv(
            os.path.join(self.input_path, "df_train.csv"), index_col=0
        )

        self.df_strain = pd.read_csv(
            os.path.join(self.input_path, "df_strain.csv"), index_col=0
        )

        self.df_grain = pd.read_csv(
            os.path.join(self.input_path, "df_grain.csv"), index_col=0
        )

        self.df_taraban = pd.read_csv(
            os.path.join(self.input_path, "df_taraban.csv"), index_col=0
        )

        self.df_glushko = pd.read_csv(
            os.path.join(self.input_path, "df_glushko.csv"), index_col=0
        )
        self.x_glushko = np.load(os.path.join(self.input_path, "ort_glushko.npz"))[
            "data"
        ]
        self.x_glushko_wf = np.array(self.df_glushko["wf"])
        self.x_glushko_img = np.array(self.df_glushko["img"])

        with open(self.input_path + "y_glushko.pkl", "rb") as f:
            self.y_glushko = pickle.load(f)

        with open(os.path.join(self.input_path, "pho_glushko.pkl"), "rb") as f:
            self.pho_glushko = pickle.load(f)

        self.phon_key = gen_pkey()

    def word_to_idx(self, word, cond=None, skip_duplicates=True):
        # TODO: Handle duplicate later

        idx = self.df_train.word.loc[self.df_train.word == word].index
        if (len(idx) == 1) or (not skip_duplicates):
            return idx.to_list()[0], cond

    def words_to_idx(self, words, conds):
        xs = list(map(self.word_to_idx, words, conds))
        idx = [x[0] for x in xs if x is not None]
        cond = [x[1] for x in xs if x is not None]
        return idx, cond

    def create_testset_from_words(self, words, conds):
        idx, cond = self.words_to_idx(words, conds)
        return self.create_testset_from_train_idx(idx, cond)

    def create_testset_from_train_idx(self, idx, cond=None):
        """Return a test set representation dictionary with word, ort, pho, sem"""
        return {
            "item": list(self.df_train.loc[idx, "word"].astype("str")),
            "cond": cond,
            "ort": tf.constant(self.np_representations["ort"][idx], dtype=tf.float32),
            "pho": tf.constant(self.np_representations["pho"][idx], dtype=tf.float32),
            "sem": tf.constant(self.np_representations["sem"][idx], dtype=tf.float32),
            "phoneme": H.get_batch_pronunciations_fast(self.np_representations["pho"][idx])
        }


    def load_all_testsets(self):

        all_testsets = ("strain", "grain")

        for testset in all_testsets:
            file = os.path.join(self.input_path, "testsets", testset + ".pkl.gz")
            self.testsets[testset] = load_testset(file)


def load_testset(file):
    with gzip.open(file, "rb") as f:
        testset = pickle.load(f)
    return testset


def save_testset(testset, file):
    with gzip.open(file, "wb") as f:
        pickle.dump(testset, f)

##### Experimentals #####
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


class BatchSampling:
    """A slim sampler for batch training"""

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.ingested_training_sample = 0
        self.current_epoch = 0
        self.current_sample = 0
        self.dynamic_corpus = dict.fromkeys(self.data.df_train.word, 0)
        np.random.seed(cfg.rng_seed)
        assert self.cfg.sample_name == "flexi_rank"

        self.x_ticks = self.cfg.n_timesteps
        self.y_ticks = self.cfg.inject_error_ticks

    def sample_generator(self, x, y):
        while True:
            sample_weights = OldSampling.get_sampling_probability(
                df_train=self.data.df_train,
                implementation=self.cfg.sample_name,
                wf_low_clip=self.cfg.wf_low_clip,
                wf_high_clip=self.cfg.wf_high_clip,
                wf_compression=self.cfg.wf_compression,
                sampling_plateau=self.cfg.sampling_plateau,
                ingested_training_sample=self.ingested_training_sample,
            )

            idx = self.data.df_train.index[sample_weights > 0]

            batch_x = [self.data.np_representations[x]] * self.x_ticks

            if type(y) is list:
                # Multi output as a list
                batch_y = [
                    [self.data.np_representations[yi]] * self.y_ticks for yi in y
                ]
            else:
                # Single output
                batch_y = [self.data.np_representations[y]] * self.y_ticks

            # Update dynamic corpus with actual exposure
            words = self.data.df_train.word.loc[idx]
            for word in words:
                self.dynamic_corpus[word] += 1

            self.ingested_training_sample += len(idx)

            yield (batch_x, batch_y, sample_weights)
