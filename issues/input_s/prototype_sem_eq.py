# %%
import os, sys
from IPython.display import clear_output
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

sys.path.append("/home/jupyter/tf/src/")
os.chdir("/home/jupyter/tf")
import meta, data_wrangling

cfg = meta.ModelConfig.from_json("models/test_sampling_speed_2/model_config.json")
data = data_wrangling.MyData()


# %% Dry run Sampling
sampler = data_wrangling.Sampling(cfg, data, debugging=True)
sampler.set_semantic_parameters(**{"g":100, "k":100})
next(sampler.sample_generator(dryrun=True)) # init generator

while sampler.debug_epoch[-1] < 100:
    print(f"Simulating epoch:{sampler.current_epoch} batch: {sampler.current_batch} | Corpus size: {sampler.debug_corpus_size[-1]}")
    next(sampler.sample_generator(dryrun=True))
    clear_output(wait=True)

print("All done.")

# %% Parse results

# Convert dict to df
df_wf = pd.DataFrame(sampler.debug_wf).transpose()
df_wf.columns = "cwf_" + df_wf.columns.astype(str)
df_wf["word"] = df_wf.index

# Convert dict to df
df_sem = pd.DataFrame(sampler.debug_sem).transpose()
df_sem.columns = "sem_" + df_sem.columns.astype(str)
df_sem["word"] = df_sem.index

# Merge
df = pd.merge(df_wf, df_sem, "left", on="word")

# Subset
df = df.loc[df.word.isin(data.df_strain.word)]
df = df.melt(id_vars=['word'],
                     var_name='epoch_label', value_name='value')

# Make new columns
df["measure"] = df.epoch_label.str.split("_", expand=True).loc[:, 0]
df["epoch"] = df.epoch_label.str.split("_", expand=True).loc[:, 1]
df.pop("epoch_label")

# Pivot
df = df.pivot_table(index=["word", "epoch"], columns="measure").reset_index()
df.columns = ['word', 'epoch', 'cumulative_frequency', 'semantic_input']

# Merge strain conditions
df = pd.merge(df, data.df_strain, "left", on='word')
df['cond'] = df.frequency + '_' + df.imageability
df.rename(columns={'wf': 'static_wf'}, inplace=True)

# %% Plot
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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

df["activation"] = df.semantic_input.apply(sigmoid)
# %%

df_mean = df.pivot_table(index="epoch", columns="cond")
df_mean["epoch"] = df_mean.index.astype(int)
df_mean.reset_index(drop=True, inplace=True)
df_mean.sort_values("epoch", inplace=True)



for measure in ["semantic_input", "activation"]:
    df_mean[f"frequency_effect_{measure}"] = df_mean[measure]["HF_HI"] + df_mean[measure]["HF_LI"] - \
        df_mean[measure]["LF_HI"] - df_mean[measure]["LF_LI"]

    df_mean[f"imageability_effect_{measure}"] = df_mean[measure]["HF_HI"] + df_mean[measure]["LF_HI"] - \
        df_mean[measure]["HF_LI"] - df_mean[measure]["LF_LI"]

    df_mean[f"fxi_interaction_{measure}"] = df_mean[measure]["LF_HI"] - df_mean[measure]["LF_LI"] - \
        df_mean[measure]["HF_HI"] + df_mean[measure]["HF_LI"]

# %%

strain_conds = ("HF_HI", "HF_LI", "LF_HI", "LF_LI")
fig = plt.figure(figsize=(15, 10))

# Semantic input
ax = fig.add_subplot(221)
ax.title.set_text("Semantic input over epoch")
for condition in strain_conds:
    ax.plot(df_mean.epoch, df_mean["semantic_input"][condition], label=condition)
ax.legend()

 # Semantic activation
ax = fig.add_subplot(222)
ax.title.set_text("Semantic activation over epoch")
for condition in strain_conds:
    ax.plot(df_mean.epoch, df_mean["activation"][condition], label=condition)
ax.legend()

# Contrasts for input
contrasts = ["frequency_effect", "imageability_effect", "fxi_interaction"]

ax = fig.add_subplot(223)
ax.title.set_text("Contrasts for input")
for contrast in contrasts:
    ax.plot(df_mean.epoch, df_mean[f"{contrast}_semantic_input"], label=contrast)
ax.legend()

# Contrasts for activation
ax = fig.add_subplot(224)
ax.title.set_text("Contrasts for activation")

for contrast in contrasts:
    ax.plot(df_mean.epoch, df_mean[f"{contrast}_activation"], label=contrast)
ax.legend()
# %%
