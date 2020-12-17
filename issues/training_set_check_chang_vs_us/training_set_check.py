
# %% Env
import sys
import pandas as pd
import numpy as np
from IPython.display import clear_output
sys.path.append("/home/jupyter/tf/dataset/")

# %% Read training file

our_dict = pd.read_csv(
    "../preprocessing/6ktraining_v2.dict",
    sep="\t",
    header=None,
    names=["word", "ort", "pho", "wf"],
    na_filter=False,  # Bug fix: incorrectly treated null as missing value in the corpus
)

chang_dict = pd.read_csv(
    '../preprocessing/6kdict',
    sep="\t",
    header=None,
    names=["word", "ort", "pho", "wf"],
    na_filter=False,  # Bug fix: incorrectly treated null as missing value in the corpus
)
chang_dict['wn_idx'] = chang_dict.index

y_wordnet = np.genfromtxt('../preprocessing/wordNet_6229.csv', delimiter=',')


# %% Export duplicates
count_chang = chang_dict.groupby("word").agg("count").reset_index()
dup_words = count_chang.loc[count_chang.wn_idx > 1, "word"]
chang_dict.loc[chang_dict.word.isin(dup_words),].to_csv("duplicate_words_chang.csv")

# %% Shape of semantics
y_wordnet.shape

