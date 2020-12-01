# %% 
from IPython import get_ipython

# %% [markdown]
# # Tidy source files to keys csv

# %%
get_ipython().run_line_magic('load_ext', 'lab_black')
import pandas as pd
import numpy as np
from IPython.display import clear_output

# Read training file
train_file = "../common/patterns/6ktraining_v2.dict"

strain_file = "../common/patterns/strain.txt"
strain_key_file = "../common/patterns/strain_key.txt"

grain_file = "../common/patterns/grain_nws.dict"
grain_key_file = "../common/patterns/grain_key.txt"

# Imageability
cortese = pd.read_csv("../common/patterns/cortese2004norms.csv", skiprows=9)
img_map = cortese[["item", "rating"]]
img_map.columns = ["word", "img"]

# Zeno norm
zeno = pd.read_csv("../common/patterns/EWFG.csv")
zeno['gr14'] = pd.to_numeric(zeno.f, errors='coerce') # Stage 14 is adult frequency
[zeno.pop(v) for v in ['sfi', 'd', 'u', 'f']]
clear_output()

# %% [markdown]
# ## Export sampling probability
# 
# 1. prob_log.txt: log compressed frequency
# 2. prob_hs04.txt: hs04 implementation
# 3. prob_jay.txt: jay's implementation
# 

# %%
# Merge Zeno and IMG into train
train = pd.read_csv(
    train_file,
    sep="\t",
    header=None,
    names=["word", "ort", "pho", "wf"],
    na_filter=False,  # Bug fix: incorrectly treated null as missing value in the corpus
)

train = pd.merge(train, zeno, on="word", how="left")

# Assume Zeno missing = 0
for x in range(14):
    variable_name = 'gr' + str(x+1)
    train[variable_name] = train[variable_name].map(lambda x: 0 if np.isnan(x) else x)

train = pd.merge(train, img_map, on="word", how="left")


# %%
strain = pd.read_csv(
    strain_file, sep="\t", header=None, names=["word", "ort", "pho", "wf"]
)

strain_key = pd.read_table(
    strain_key_file,
    header=None,
    delim_whitespace=True,
    names=["word", "frequency", "pho_consistency", "imageability"],
)

strain = pd.merge(strain, strain_key)
strain = pd.merge(strain, img_map, on="word", how="left")
strain.sample(5)


# %%
strain.groupby("frequency").mean()


# %%
grain = pd.read_csv(
    grain_file,
    sep='\t',
    header=None,
    names=['word', 'ort', 'pho_large', 'pho_small']
)

grain_key = pd.read_table(
    grain_key_file,
    header=None,
    delim_whitespace=True,
    names=['word', 'condition']
)

grain_key['condition'] = np.where(
    grain_key['condition'] == 'critical', 'ambiguous', 'unambiguous'
)

grain = pd.merge(grain, grain_key)

grain['img'] = 0
grain['wf'] = 0
grain.sample(5)


# %%
taraban = pd.read_csv('../common/patterns/taraban.csv')
taraban.columns = ['id', 'cond', 'word', 'ort', 'pho', 'wf']
taraban = pd.merge(taraban, img_map, on='word', how='left')
taraban.sample(5)


# %%
glushko = pd.read_csv('../common/patterns/glushko_nonword.csv')
glushko.columns = ['id', 'cond', 'word', 'pho', 'ort']

glushko['img'] = 0
glushko['wf'] = 0
glushko.sample(5)

# %% [markdown]
# ### Check raw data integrity

# %%
# Check all represtation follow 14 ort, 10 pho format
assert all([len(x) == 14 for x in train.ort])
assert all([len(x) == 14 for x in strain.ort])
assert all([len(x) == 14 for x in grain.ort])
assert all([len(x) == 14 for x in taraban.ort])
assert all([len(x) == 14 for x in glushko.ort])

assert all([len(x) == 10 for x in train.pho])
assert all([len(x) == 10 for x in strain.pho])
assert all([len(x) == 10 for x in grain.pho_small])
assert all([len(x) == 10 for x in grain.pho_large])
assert all([len(x) == 10 for x in taraban.pho])

from ast import literal_eval
for pho in glushko.pho:
    ps = literal_eval(pho)
    for p in ps:
        assert len(p) == 10

# Check all fufill trim_ort criteria
locs = [0, 11, 12, 13]

for l in locs:
    assert all([x == '_' for x in train.ort.str.get(l)])
    assert all([x == '_' for x in strain.ort.str.get(l)])
    assert all([x == '_' for x in grain.ort.str.get(l)])
    assert all([x == '_' for x in taraban.ort.str.get(l)])
    assert all([x == '_' for x in glushko.ort.str.get(l)])

# No missing data in critical variables
assert sum(train.ort.isna()) == 0
assert sum(train.pho.isna()) == 0
assert sum(train.wf.isna()) == 0

assert sum(strain.ort.isna()) == 0
assert sum(strain.pho.isna()) == 0
assert sum(strain.wf.isna()) == 0

assert sum(grain.ort.isna()) == 0
assert sum(grain.pho_small.isna()) == 0
assert sum(grain.pho_large.isna()) == 0

assert sum(taraban.ort.isna()) == 0
assert sum(taraban.pho.isna()) == 0

assert sum(glushko.ort.isna()) == 0
assert sum(glushko.pho.isna()) == 0


# %%
def trim_ort(t):
    # The first bit and last 3 bits are empty in this source dataset (6ktraining.dict)
    t['ort'] = t.ort.apply(lambda x: x[1:11])
    return t


df_train = trim_ort(train)
df_strain = trim_ort(strain)
df_grain = trim_ort(grain)
df_taraban = trim_ort(taraban)
df_glushko = trim_ort(glushko)

# %% [markdown]
# # Imageability missing data replacement

# %%
def chk_missing(df, var):
    print(
        '{} missing in {}: {}/{}'.format(
            var, df, sum(globals()[df][var].isna()), len((globals()[df]))
        )
    )

chk_missing('df_train', 'img')
chk_missing('df_strain', 'img')
chk_missing('df_grain', 'img')
chk_missing('df_taraban', 'img')
chk_missing('df_glushko', 'img')


# %%
# Fill missing value to mean img rating
mean_img = df_train.img.mean()
df_train['img'] = df_train.img.fillna(mean_img)

# Fill missing value to condition mean img rating
mean_strain_hi_img = df_strain.loc[df_strain.imageability == "HI", 'img'].mean()
mean_strain_lo_img = df_strain.loc[df_strain.imageability == "LI", 'img'].mean()

df_strain.loc[df_strain.imageability == "HI",
              "img"] = df_strain.loc[df_strain.imageability == "HI",
                                     "img"].fillna(mean_strain_hi_img)

df_strain.loc[df_strain.imageability == "LI",
              "img"] = df_strain.loc[df_strain.imageability == "LI",
                                     "img"].fillna(mean_strain_lo_img)

# Since taraban do not maniputate img, just replace by training set mean
df_taraban['img'] = df_taraban.img.fillna(mean_img)


# %%
df_train.to_csv('../common/input/df_train.csv')
df_strain.to_csv('../common/input/df_strain.csv')
df_grain.to_csv('../common/input/df_grain.csv')
df_taraban.to_csv('../common/input/df_taraban.csv')
df_glushko.to_csv('../common/input/df_glushko.csv')

# %% [markdown]
# # Encode input and output

# %%
# Encode orthographic representation
def ort2bin(o_col, trimMode=True, verbose=True):
    # Replicating support.py (o_char)
    # This function wrap tokenizer.texts_to_matrix to fit on multiple
    # independent slot-based input
    # i.e. one-hot encoding per each slot with independent dictionary

    from tensorflow.keras.preprocessing.text import Tokenizer

    nSlot = len(o_col[0])
    nWord = len(o_col)

    slotData = nWord * [None]
    binData = pd.DataFrame()

    for slotId in range(nSlot):
        for wordId in range(nWord):
            slotData[wordId] = o_col[wordId][slotId]

        t = Tokenizer(filters='', lower=False)
        t.fit_on_texts(slotData)
        seqData = t.texts_to_sequences(
            slotData
        )  # Maybe just use sequence data later

        # Triming first bit in each slot
        if trimMode == True:
            tmp = t.texts_to_matrix(slotData)
            thisSlotBinData = tmp[:, 1::
                                 ]  # Remove the first bit which indicate a separate slot (probably useful in recurrent network)
        elif trimMode == False:
            thisSlotBinData = t.texts_to_matrix(slotData)

        # Print dictionary details
        if verbose == True:
            print(
                'Slot {} (n = {}, unique token = {}) {} \n'.format(
                    slotId, t.document_count, len(t.word_index.items()),
                    t.word_docs
                )
            )

        # Put binary data into a dataframe
        binData = pd.concat(
            [binData, pd.DataFrame(thisSlotBinData)], axis=1, ignore_index=True
        )
        
    return binData

def ort2bin_v2(o_col):
    # Use tokenizer instead to acheive same thing, but with extra zeros columns
    # Will be useful for letter level recurrent model
    from tensorflow.keras.preprocessing.text import Tokenizer
    t = Tokenizer(filters='', lower=False, char_level=True)
    t.fit_on_texts(o_col)
    print('dictionary:', t.word_index)
    return t.texts_to_matrix(o_col)


# Merge all 3 ortho representation
all_word = pd.concat(
    [
        df_train.word, df_strain.word, df_grain.word, df_taraban.word,
        df_glushko.word
    ],
    ignore_index=True
)

all_ort = pd.concat(
    [df_train.ort, df_strain.ort, df_grain.ort, df_taraban.ort, df_glushko.ort],
    ignore_index=True
)

# Encoding orthographic representation
all_ort_bin = ort2bin(all_ort, verbose=True)


# %%
splitId_strain = len(df_train)
splitId_grain = splitId_strain + len(df_strain)
splitId_taraban = splitId_grain + len(df_grain)
splitId_glushko = splitId_taraban + len(df_taraban)

x_train = np.array(all_ort_bin[0:splitId_strain])
x_strain = np.array(all_ort_bin[splitId_strain:splitId_grain])
x_grain = np.array(all_ort_bin[splitId_grain:splitId_taraban])
x_taraban = np.array(all_ort_bin[splitId_taraban:splitId_glushko])
x_glushko = np.array(all_ort_bin[splitId_glushko::])

# Save to disk
np.savez_compressed('../common/input/x_train.npz', data=x_train)
np.savez_compressed('../common/input/x_strain.npz', data=x_strain)
np.savez_compressed('../common/input/x_grain.npz', data=x_grain)
np.savez_compressed('../common/input/x_taraban.npz', data=x_taraban)
np.savez_compressed('../common/input/x_glushko.npz', data=x_glushko)

print('==========Orthographic representation==========')
print('all shape:', all_ort_bin.shape)
print('x_train shape:', x_train.shape)
print('x_strain shape:', x_strain.shape)
print('x_grain shape:', x_grain.shape)
print('x_taraban shape:', x_taraban.shape)
print('x_glushko shape:', x_glushko.shape)


# %%
def pho2bin_v2(p_col, p_key):
    # Vectorize for performance (that no one ask for... )
    binLength = len(p_key['_'])
    nPhoChar = len(p_col[0])

    p_output = np.empty([len(p_col), binLength * nPhoChar])

    for slot in range(len(p_col[0])):
        slotSeries = p_col.str.slice(start=slot, stop=slot + 1)
        out = slotSeries.map(p_key).to_list()
        p_output[:, range(slot * 25, (slot + 1) * 25)] = out
    return p_output


from data_wrangling import gen_pkey
phon_key = gen_pkey()
y_train = pho2bin_v2(train.pho, phon_key)
y_strain = pho2bin_v2(strain.pho, phon_key)
y_large_grain = pho2bin_v2(grain.pho_large, phon_key)
y_small_grain = pho2bin_v2(grain.pho_small, phon_key)
y_taraban = pho2bin_v2(taraban.pho, phon_key)

# Save to disk
np.savez_compressed('../common/input/y_train.npz', data=y_train)
np.savez_compressed('../common/input/y_strain.npz', data=y_strain)
np.savez_compressed('../common/input/y_large_grain.npz', data=y_large_grain)
np.savez_compressed('../common/input/y_small_grain.npz', data=y_small_grain)
np.savez_compressed('../common/input/y_taraban.npz', data=y_taraban)

print('\n==========Phonological representation==========')
print(len(phon_key), ' phonemes: ', phon_key.keys())
print('y_train shape:', y_train.shape)
print('y_strain shape:', y_strain.shape)
print('y_large_grain shape:', y_large_grain.shape)
print('y_small_grain shape:', y_small_grain.shape)
print('y_taraban shape:', y_taraban.shape)

# %% [markdown]
# ### Decoding check

# %%
from evaluate import get_all_pronunciations_fast as gapf
assert all(gapf(y_train, phon_key) == df_train.pho)
assert all(gapf(y_strain, phon_key) == df_strain.pho)
assert all(gapf(y_large_grain, phon_key) == df_grain.pho_large)
assert all(gapf(y_small_grain, phon_key) == df_grain.pho_small)
assert all(gapf(y_taraban, phon_key) == df_taraban.pho)

# %% [markdown]
# ## Special format for Glushko PHO (due to multiple correct answer with different length)

# %%
import ast, pickle

# Glushko pho dictionary
pho_glushko = {
    x: ast.literal_eval(df_glushko.loc[i, 'pho'])
    for i, x in enumerate(df_glushko.word)
}

with open('../common/input/pho_glushko.pkl', 'wb') as f:
    pickle.dump(pho_glushko, f)

# Glushko one-hot encoded output dictionary
y_glushko = {}
for k, v in pho_glushko.items():
    ys = []
    for pho in v:
        y = []
        for char in pho:
            y += phon_key[char]
        ys.append(y)
    y_glushko[k] = ys

with open('../common/input/y_glushko.pkl', 'wb') as f:
    pickle.dump(y_glushko, f)

print('y_glushko dimension: {}'.format(len(y_glushko['beed'][0])))

# %% [markdown]
# # Testing and evaluating new sampling probability

# %%
import pandas as pd
import numpy as np
df_train = pd.read_csv('../common/input/df_train.csv', index_col=0)

# Plot sampling conversion graph
import matplotlib.pyplot as plt
import data_wrangling

plot_f = df_train.sort_values('wf')

fig, ax = plt.subplots(facecolor="w")
line1, = ax.plot(plot_f.wf, data_wrangling.get_sampling_probability(plot_f, "log"), label='Log')
line2, = ax.plot(plot_f.wf, data_wrangling.get_sampling_probability(plot_f, "hs04"), label='HS04')
line3, = ax.plot(plot_f.wf, data_wrangling.get_sampling_probability(plot_f, "jay"), label='JAY')

ax.legend(loc='lower right')
plt.xlabel('Word frequency')
# plt.xlim((0, 200))
# plt.ylim((0, .0006))
plt.ylabel('Sampling probability')
# plt.xlim([0,100])
plt.title('Tested sampling p vs. word frequency')
plt.show()


