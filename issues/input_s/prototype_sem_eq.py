# %%
import os
import sys
from IPython.display import clear_output
import pandas as pd
import altair as alt

sys.path.append("/home/jupyter/tf/src/")
os.chdir("/home/jupyter/tf")
import meta, data_wrangling

from importlib import reload
reload(data_wrangling)

cfg = meta.ModelConfig.from_json("models/test_sampling_speed_2/model_config.json")
data = data_wrangling.MyData()


# %% Dry run Sampling
sampler = data_wrangling.Sampling(cfg, data, debugging=True)
sampler.set_semantic_parameters(**{"g":5, "k":100})
next(sampler.sample_generator()) # init generator

while sampler.debug_epoch[-1] <= 100:
    print(f"Simulating epoch:{sampler.debug_epoch[-1]}")
    next(sampler.sample_generator())
    clear_output(wait=True)

print("All done.")


# %% Parse results
df = pd.merge(sampler.debug_wf.reset_index(), sampler.debug_sem.reset_index(), "left", on="word")
df = df.loc[df.word.isin(data.df_strain.word)]
df = df.melt(id_vars=['word'],
                     var_name='epoch_label', value_name='value')

df["measure"] = df.epoch_label.str.split("_", expand=True).loc[:, 0]
df["epoch"] = df.epoch_label.str.split("_", expand=True).loc[:, 3]
df.pop("epoch_label")

df = df.pivot_table(index=["word", "epoch"], columns="measure").reset_index()
df.columns = ['word', 'epoch', 'semantic_input', 'cumulative_frequency']

# Merge strain conditions
df = pd.merge(df, data.df_strain, "left", on='word')
df['cond'] = df.frequency + '_' + df.imageability
df.rename(columns={'wf': 'static_wf'}, inplace=True)


# %%
plot_sem = alt.Chart(df).mark_line().encode(
    x="epoch:Q",
    y="mean(semantic_input):Q",
    color="cond:N",
)

plot_wf = alt.Chart(df).mark_line().encode(
    x="epoch:Q",
    y="mean(cumulative_frequency):Q",
    color="cond:N",
)

plot_wf | plot_sem


# %%
