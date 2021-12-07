# This script contain a set of custom functions for managing representations and sampling

import pickle, gzip, os
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
tf_root = os.environ.get("TF_ROOT")


def gen_pkey(key_file=None) -> dict:
    """Read phonological patterns from the mapping file.
    See Harm & Seidenberg PDF file
    """
    if key_file is None:
        mapping = os.path.join(tf_root, "dataset", "mappingv2.txt")

    mapping = pd.read_table(mapping, header=None, delim_whitespace=True)
    mapping_dict = mapping.set_index(0).T.to_dict("list")
    return mapping_dict


def get_pronunciation_fast(act: np.array, phon_key: dict = None) -> str:
    if phon_key is None:
        phon_key = gen_pkey()
    phonemes = list(phon_key.keys())
    act10 = np.tile([v for k, v in phon_key.items()], 10)

    d = np.abs(act10 - act)
    d_mat = np.reshape(d, (38, 10, 25))
    sumd_mat = np.squeeze(np.sum(d_mat, 2))
    map_idx = np.argmin(sumd_mat, 0)
    out = str()
    for x in map_idx:
        out += phonemes[x]
    return out


def get_batch_pronunciations_fast(act: np.array, phon_key: dict = None) -> np.array:
    """Get a batch of pronunciations from activations."""
    if phon_key is None:
        phon_key = gen_pkey()
    return np.apply_along_axis(get_pronunciation_fast, 1, act, phon_key)


class MyData:
    """
    This object load all clean data from disk (both training set and testing sets)
    Also calculate sampling_p according to cfg.sample_name setting
    """

    def __init__(self, input_path="dataset"):

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

        with open(os.path.join(self.input_path, "y_glushko.pkl"), "rb") as f:
            self.y_glushko = pickle.load(f)

        with open(os.path.join(self.input_path, "pho_glushko.pkl"), "rb") as f:
            self.pho_glushko = pickle.load(f)

        self.phon_key = gen_pkey(p_file=os.path.join(input_path, "mappingv2.txt"))

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
        """Create a testset dictionary package with a list of words"""
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
            "phoneme": get_batch_pronunciations_fast(
                self.np_representations["pho"][idx]
            ),
        }

    def load_all_testsets(self):

        all_testsets = ("strain", "grain")

        for testset in all_testsets:
            file = os.path.join(self.input_path, "testsets", testset + ".pkl.gz")
            self.testsets[testset] = load_testset(file)


def load_testset(file):
    """Load testset from pickled file"""
    tf_root = os.environ.get("TF_ROOT")
    try:
        with gzip.open(file, "rb") as f:
            testset = pickle.load(f)
    except Exception:
        _maybe_file = os.path.join(tf_root, "dataset", "testsets", f"{file}.pkl.gz")
        with gzip.open(_maybe_file, "rb") as f:
            testset = pickle.load(f)
    return testset


def save_testset(testset, file):
    with gzip.open(file, "wb") as f:
        pickle.dump(testset, f)
