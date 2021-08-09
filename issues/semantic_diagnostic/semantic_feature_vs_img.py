#%% 
import data_wrangling
import pandas as pd
import numpy as np
import altair as alt
# %% Load test sets from file
d = data_wrangling.MyData()
d.load_testsets()

# %% A function to get number of features
def get_n_features(word):
    if word in d.testsets['train']['item']:
        idx = d.testsets['train']['item'].index(word)
        semantic_vector = d.testsets['train']['sem'][idx,:]
        return(sum(semantic_vector)) 



# %% Parse Cortese
cortese = pd.read_csv(
    "../preprocessing/cortese2004norms.csv", skiprows=9, na_filter=False
)
cortese = cortese[["item", "rating"]]
cortese = cortese.rename({"item": "word", "rating":"img"}, axis=1)
cortese["n_features"] = cortese.word.apply(get_n_features)

cortese = cortese.merge(d.df_train[['word', 'wf']])

# %% Also get probability by different implementation
cortese["p_hs04"] = data_wrangling.Sampling.get_sampling_probability(cortese, "hs04")
cortese["p_zevin"] = data_wrangling.Sampling.get_sampling_probability(cortese, "jay")
cortese["log_wf"] = data_wrangling.Sampling.get_sampling_probability(cortese, "log")


# %% Plot IMG vs. n_features
nf = alt.Chart(cortese).mark_point().encode(x="img", y="n_features", tooltip=["word", "img"])
nf += nf.transform_regression('img', 'n_features').mark_line(color="red")
nf.properties(title = cortese.corr())
# %% Pearson's r
cortese.corr()

# %%
p = alt.Chart(cortese).mark_point().encode(x="img", y="p_hs04", tooltip=["word", "p_hs04"])
p + p.transform_regression('img', 'p_hs04').mark_line(color="red")

# %%
