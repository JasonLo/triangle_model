#%% Environment
import gzip
import pickle

import numpy as np

import data_wrangling

data = data_wrangling.MyData()
# %% Helper functions


def sem_to_idx(idx):
    """Convert one-hot sem to actiavtion index"""
    sem = data.sem_train[
        idx,
    ]
    (activated_sem_nodes,) = np.where(sem == 1.0)
    activated_sem_nodes = activated_sem_nodes.tolist()

    # plural_nodes = [21, 22, 32]
    # 21 from cut
    # 22 from jam
    # 32 from shrimp

    # Remove plural semantic node
    # On 2nd thought, tenses, plural within semantic... probably don't need to remove
    # for node in activated_sem_nodes:
    #     if node in plural_nodes:
    #         activated_sem_nodes.remove(node)

    return activated_sem_nodes


def is_homophone(word_idx):
    """Ugly one use function for finding homophone"""
    # Get phonology
    word_pho = data.df_train.loc[word_idx, "pho"]
    same_phos = data.df_train.loc[data.df_train.pho == word_pho, "pho"]

    # Get a list of word index with the same phonology
    same_pho_idx = same_phos.index.to_list()

    if len(same_pho_idx) == 1:
        # Not a homophone if the corpus has only one record of this phonology
        return False
    else:
        # Get semantics (in index format)
        sems = [sem_to_idx(id) for id in same_pho_idx]

        # Count unique semantic representation
        count_unique_sem = 1
        for sem in sems:
            if sems[0] != sem:
                count_unique_sem += 1

        if count_unique_sem > 1:
            # One pho, multi sem is a homophone
            return True
        else:
            # One pho, one sem... multi words/ort... weird... but still not homophone
            return False


# %% Create a table for manual exam
table_homephone = data.df_train.copy()
table_homephone["idx"] = table_homephone.index
table_homephone["sem_activate_node"] = table_homephone.idx.apply(sem_to_idx)
table_homephone["is_homophone"] = table_homephone.idx.apply(is_homophone)
table_homephone.sort_values("pho", inplace=True)
table_homephone.to_csv("homophone_check.csv")

# %% Create homophone related testset
testset_homophone_idx = table_homephone.loc[
    table_homephone.is_homophone, "idx"
].to_list()

testset_non_homophone_idx = table_homephone.loc[
    table_homephone.is_homophone == False, "idx"
].to_list()


testset_homophone = data.create_testset_from_train_idx(testset_homophone_idx)
testset_non_homophone = data.create_testset_from_train_idx(testset_non_homophone_idx)

# %% Save

with gzip.open("dataset/testsets/homophone.pkl.gz", "wb") as f:
    pickle.dump(testset_homophone, f)

with gzip.open("dataset/testsets/non_homophone.pkl.gz", "wb") as f:
    pickle.dump(testset_non_homophone, f)

# %% Load example

with gzip.open("dataset/testsets/homophone.pkl.gz", "rb") as f:
    testset_homophone = pickle.load(f)

with gzip.open("dataset/testsets/non_homophone.pkl.gz", "rb") as f:
    testset_non_homophone = pickle.load(f)


# %% Test packaged MyData() class
from importlib import reload
reload(data_wrangling)
data = data_wrangling.MyData()
data.testsets["homophone"]["word"]
