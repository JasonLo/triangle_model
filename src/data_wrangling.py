# This script contain a set of custom functions for managing representations and sampling

import pickle, gzip, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from dotenv import load_dotenv
from typing import List

load_dotenv()
tf_root = os.environ.get("TF_ROOT")

# Preprocessing or convienience functions for looking at raw data


def get_duplicates(df: pd.DataFrame, group: str) -> dict:
    """Analyze duplicates word and thier counts in a dataframe."""
    duplicates = df.groupby([group]).size().reset_index(name="counts")
    duplicates = duplicates[duplicates.counts > 1]
    return dict(zip(duplicates[group], duplicates.counts))


def get_index_of_one(rep: list) -> list:
    """Return the index of 1 values in a list.
    For compressing a dense representation into a sparse one.

    Arguments:
        representation: List of values. (Assume 1d list)
    """
    return [i for i, x in enumerate(rep) if x == 1]


def get_used_slots(rep: list) -> set:
    """Get the index of non empty slots in a list of representation.

    Arguments:
        rep: List of representation
    """
    return {i for i, x in enumerate(rep) if x != "_"}


def remove_slots(rep: list, remove_slots: list) -> list:
    """Removing a slot from a list of representation.

    Arguments:
        rep: List of representation
        slot: Slot to remove
    """
    l = [len(x) for x in rep]
    assert len(set(l)) == 1  # Check that all the representation have the same length

    slots = range(len(rep[0]))
    return ["".join(x[s] for s in slots if s not in remove_slots) for x in rep]


def trim_unused_slots(rep: list) -> list:
    """Trimming unused slot in a list of representation.

    Arguments:
        rep: List of representation
    """
    slots = range(len(rep[0]))
    print(f"We have these slots: {list(slots)}")

    slot_unique_set = {slot: set([x[slot] for x in rep]) for slot in slots}
    unused_slots = set(
        [slot for slot, slot_set in slot_unique_set.items() if len(slot_set) == 1]
    )
    print(f"Removing unused slots: {list(unused_slots)}")

    return remove_slots(rep, unused_slots)


def one_hot_ort_slot(slot_data: List[str], tokenizer: Tokenizer = None) -> np.array:
    """One-hot encode orthographic representation of one orthographic slot."""

    if tokenizer is None:
        tokenizer = Tokenizer(filters="", lower=False)
        tokenizer.fit_on_texts(slot_data)
    bin_data = tokenizer.texts_to_matrix(slot_data)
    print(f"Token count: {tokenizer.word_docs}")
    return bin_data[:, 1:], tokenizer  # remove first column (padding)


def ort_to_binary(ort: List[str], tokenizers: list = None) -> np.array:
    """Convert orthographic representation to binary.

    Replicating Jason Zevin's support.py (o_char), It one-hot encode each letter with independent dictionary in each slot.
    Finally, trimming the unused units.

    Also export the tokenizers for each slot for later use.
    """
    n_slots = len(ort[0])
    bin_slot_data = []

    if tokenizers is None:
        tokenizers = [None] * n_slots

    for slot in range(n_slots):
        this_slot_bin_data, this_tokenizer = one_hot_ort_slot(
            [x[slot] for x in ort], tokenizers[slot]
        )
        bin_slot_data.append(this_slot_bin_data)
        tokenizers[slot] = this_tokenizer

    return np.concatenate(bin_slot_data, axis=1), tokenizers


def check_oov_ort(tokenizers: List[Tokenizer], ort: List[str]) -> set:
    """Check if there are any out of vocabulary words in the orthographic representation."""
    oov_pool = {}

    for slot in range(10):
        print(f"At slot {slot}:")
        tokenizer_o = set(tokenizers[slot].word_index.keys())
        print(f"    tokenizer: {tokenizer_o}")
        unique_o = {o[slot] for o in ort}
        print(f"    data: {unique_o}")
        oov = unique_o.difference(tokenizer_o)  # Out of vocabulary.
        print(f"    diff: {oov}")

        oov_pool.update(oov)

    return oov_pool


def get_homophones(train: pd.DataFrame) -> List[str]:
    """Create a list of homophones."""
    df = train.groupby("pho").count().reset_index()
    return df.loc[df.word > 1, "pho"].to_list()


def gen_pkey(key_file: str = None) -> dict:
    """Read phonological patterns from the mapping file.
    See Harm & Seidenberg PDF file
    """
    if key_file is None:
        key_file = os.path.join(tf_root, "dataset", "mappingv2.txt")

    mapping = pd.read_table(key_file, header=None, delim_whitespace=True)
    mapping_dict = mapping.set_index(0).T.to_dict("list")
    return mapping_dict


def pho_to_binary(pho: List[str], mapping: dict = None) -> np.array:
    """Convert phonological representation to binary."""
    if mapping is None:
        mapping = gen_pkey()

    bin_length = len(mapping["_"])
    n_slots = len(pho[0])

    # Preallocate the entire ouput
    binary_pho = np.empty([len(pho), bin_length * n_slots])

    for slot in range(n_slots):
        slot_data = pho.str.slice(start=slot, stop=slot + 1)
        out = slot_data.map(mapping).to_list()
        binary_pho[:, range(slot * 25, (slot + 1) * 25)] = out
    return binary_pho


def get_pronunciation_fast(act: np.array, phon_key: dict = None) -> str:
    """Get a pronunciation from activations. Optimized for compute speed."""
    if phon_key is None:
        phon_key = gen_pkey()
    phonemes = list(phon_key.keys())

    n_slots = len(act) // 25
    act10 = np.tile([v for v in phon_key.values()], n_slots)

    d = np.abs(act10 - act)
    d_mat = np.reshape(d, (38, n_slots, 25))
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


# class MyData:  # Obsolete
#     """
#     This object load all clean data from disk (both training set and testing sets)
#     Also calculate sampling_p according to cfg.sample_name setting
#     """

#     def __init__(self, input_path="dataset"):

#         self.input_path = os.path.join(tf_root, input_path)

#         # init an empty testset dict for new testset format
#         # first level: testset name
#         # second level (in each testset): word, ort, pho, sem
#         self.testsets = {}
#         self.load_all_testsets()

#         self.np_representations = {
#             "ort": np.load(os.path.join(self.input_path, "ort_train.npz"))["data"],
#             "pho": np.load(os.path.join(self.input_path, "pho_train.npz"))["data"],
#             "sem": np.load(os.path.join(self.input_path, "sem_train.npz"))["data"],
#         }

#         self.df_train = pd.read_csv(
#             os.path.join(self.input_path, "df_train.csv"), index_col=0
#         )

#         self.df_strain = pd.read_csv(
#             os.path.join(self.input_path, "df_strain.csv"), index_col=0
#         )

#         self.df_grain = pd.read_csv(
#             os.path.join(self.input_path, "df_grain.csv"), index_col=0
#         )

#         self.df_taraban = pd.read_csv(
#             os.path.join(self.input_path, "df_taraban.csv"), index_col=0
#         )

#         self.df_glushko = pd.read_csv(
#             os.path.join(self.input_path, "df_glushko.csv"), index_col=0
#         )
#         self.x_glushko = np.load(os.path.join(self.input_path, "ort_glushko.npz"))[
#             "data"
#         ]
#         self.x_glushko_wf = np.array(self.df_glushko["wf"])
#         self.x_glushko_img = np.array(self.df_glushko["img"])

#         with open(os.path.join(self.input_path, "y_glushko.pkl"), "rb") as f:
#             self.y_glushko = pickle.load(f)

#         with open(os.path.join(self.input_path, "pho_glushko.pkl"), "rb") as f:
#             self.pho_glushko = pickle.load(f)

#         self.phon_key = gen_pkey()

#     def word_to_idx(self, word, cond=None, skip_duplicates=True):
#         # TODO: Handle duplicate later

#         idx = self.df_train.word.loc[self.df_train.word == word].index
#         if (len(idx) == 1) or (not skip_duplicates):
#             return idx.to_list()[0], cond

#     def words_to_idx(self, words, conds):
#         xs = list(map(self.word_to_idx, words, conds))
#         idx = [x[0] for x in xs if x is not None]
#         cond = [x[1] for x in xs if x is not None]
#         return idx, cond

#     def create_testset_from_words(self, words, conds):
#         """Create a testset dictionary package with a list of words"""
#         idx, cond = self.words_to_idx(words, conds)
#         return self.create_testset_from_train_idx(idx, cond)

#     def create_testset_from_train_idx(self, idx, cond=None):
#         """Return a test set representation dictionary with word, ort, pho, sem"""
#         return {
#             "item": list(self.df_train.loc[idx, "word"].astype("str")),
#             "cond": cond,
#             "ort": tf.constant(self.np_representations["ort"][idx], dtype=tf.float32),
#             "pho": tf.constant(self.np_representations["pho"][idx], dtype=tf.float32),
#             "sem": tf.constant(self.np_representations["sem"][idx], dtype=tf.float32),
#             "phoneme": get_batch_pronunciations_fast(
#                 self.np_representations["pho"][idx]
#             ),
#         }

#     def load_all_testsets(self):

#         all_testsets = ("strain", "grain")

#         for testset in all_testsets:
#             file = os.path.join(self.input_path, "testsets", testset + ".pkl.gz")
#             self.testsets[testset] = load_testset(file)


def load_testset(file):
    """Load testset from pickled file."""

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


def create_testset_from_words(words, conds=None) -> dict:
    """Create a testset dictionary package with a list of words."""

    train = load_testset("dataset/train.pkl.gz")
    sel_idx = [train["item"].index(word) for word in words if word in train["item"]]
    print(f"{len(sel_idx)} out of {len(words)} words in new train set")

    return {
        "item": [train["item"][idx] for idx in sel_idx],
        "phoneme": [train["phoneme"][idx] for idx in sel_idx],
        "graphem": [train["graphem"][idx] for idx in sel_idx],
        "ort": tf.gather(train["ort"], sel_idx),
        "pho": tf.gather(train["pho"], sel_idx),
        "sem": tf.gather(train["sem"], sel_idx),
        "cond": conds,
    }
