import numpy as np
import pandas as pd

def gen_pkey(p_file="dataset/mappingv2.txt"):
    """Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict("list")
    return m_dict


def get_pronunciation_fast(act, phon_key=None):
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


def get_batch_pronunciations_fast(act, phon_key=None):
    if phon_key is None:
        phon_key = gen_pkey()
    return np.apply_along_axis(get_pronunciation_fast, 1, act, phon_key)