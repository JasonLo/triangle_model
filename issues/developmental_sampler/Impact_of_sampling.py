# %% Environment
import importlib
import os
import altair as alt
import pandas as pd
from IPython.display import clear_output

os.chdir("/home/jupyter/tf")
from src import data_wrangling, meta

alt.data_transformers.disable_max_rows()
importlib.reload(data_wrangling)

cfg = meta.ModelConfig(json_file='models/test_sampling_speed_hparam/model_config.json')
data = data_wrangling.MyData()
working_directory = "issues/developmental_sampler/dynamic_frequency/"


def dry_run_sampler(sample_name, cfg, output_folder=working_directory):
    cfg.sample_name = sample_name
    sampler = data_wrangling.Sampling(cfg, data, debugging=True)
    next(sampler.sample_generator())
    while sampler.debug_log_epoch[-1] <= 100:
        print(f"Running {sample_name} epoch:{sampler.debug_log_epoch[-1]}")
        next(sampler.sample_generator())
        clear_output(wait=True)

    df_corpus_size = pd.DataFrame({"epoch": sampler.debug_log_epoch,
                                   "corpus_size": sampler.debug_log_corpus_size})
    df_corpus_size.to_csv(
        f"{output_folder}{sample_name}_corpus_size.csv")
    sampler.debug_log_dynamic_wf.to_csv(
        f"{output_folder}{sample_name}_dynamic_corpus.csv")
    print("All done.")


dry_run_sampler('developmental_rank_frequency', cfg)

# %% Show frequency statistics

data.df_train['clip_wf_30000'] = data.df_train.wf.clip(0, 30000)

wf_plot = alt.Chart(data.df_train).mark_bar().encode(
    y='count(clip_wf_30000):Q',
    x=alt.X('clip_wf_30000', bin=alt.Bin(maxbins=50)),
).interactive().properties(title="Histogram of WSJ frequncy with top-end squashing")

wf_plot.save(f"{working_directory}histogram_wf.html")


# %% Zoom-in 1000
df_1000 = data.df_train.loc[data.df_train.wf <= 1000,]
wf_plot_1000 = alt.Chart(df_1000).mark_bar().encode(
    y='count(clip_wf_30000):Q',
    x=alt.X('clip_wf_30000', bin=alt.Bin(maxbins=50)),
).interactive().properties(title="Histogram of WSJ frequncy with top-end squashing")


wf_plot_1000.wf_plot.save(f"{working_directory}histogram_wf_1000.html")

# %% Dry run and log dynamic corpus
[dry_run_sampler(sample_name, cfg)
 for sample_name in ['experimental', 'wf_linear_cutoff']]


# %% Visualize corpus size
implementations = ["chang", "experimental", "hs04", "jay", "wf_linear_cutoff"]

# Make combined df
df = pd.DataFrame()
for i in implementations:
    tmp_df = pd.read_csv(f"{working_directory}{i}_corpus_size.csv")
    tmp_df['sample_name'] = i
    df = pd.concat([df, tmp_df], axis=0)


plot_dcs = alt.Chart(df).mark_line().encode(
    x="epoch:Q",
    y=alt.Y("corpus_size:Q", title="Cumulative corpus size"),
    color="sample_name"
).properties(title="Cumulative corpus size").interactive()


plot_dcs.save(f'{working_directory}corpus_size.html')
plot_dcs

# %% Visualize dynamic corpus

df = pd.DataFrame()
for i in implementations:
    tmp_df = pd.read_csv(f"{working_directory}{i}_dynamic_corpus.csv")
    tmp_df['sample_name'] = i
    df = pd.concat([df, tmp_df], axis=0)


# Only select Strain items
sel_df = df.loc[df.word.isin(data.df_strain.word)]
sel_df = sel_df.melt(id_vars=['word', 'sample_name'],
                     var_name='epoch_label', value_name='wf')
sel_df['epoch'] = sel_df.epoch_label.apply(lambda x: int(x[12::]) - 1)
sel_df.rename(columns={'wf': 'dynamic_wf'}, inplace=True)

# Down-sample epoch
sel_epoch = [0, 1, 3, 5, 7, 10, 30, 40, 50, 80, 100]
sel_df = sel_df.loc[sel_df.epoch.isin(sel_epoch), ]

df_strain = pd.merge(sel_df, data.df_strain, "left", on='word')
df_strain['cond'] = df_strain.frequency + '_' + df_strain.pho_consistency
df_strain.rename(columns={'wf': 'static_wf'}, inplace=True)

# Dynamic frequncy in Strain by condition
plot_corpus_epoch = alt.Chart(df_strain).mark_line().encode(
    x="epoch:Q",
    y=alt.Y("mean(dynamic_wf):Q", title="Mean cumulative frequency"),
    color="sample_name:N",
    column="frequency:N",
    row="pho_consistency:N"
).interactive().properties(title="Mean cumulative frequency in each condition in Strain")

plot_corpus_epoch.save(
    f'{working_directory}strain_cumulative_frequency.html')

plot_corpus_epoch


# %% Relationship between actual corpus frequency and dynamic frequency
input_dropdown = alt.binding_select(options=sel_epoch)
selection_epoch = alt.selection_single(
    fields=['epoch'], bind=input_dropdown, init={'epoch': 100})
selection_sampling = alt.selection_multi(fields=['sample_name'], bind='legend')

plot_ff = alt.Chart(df_strain).mark_point().encode(
    y=alt.Y('dynamic_wf', title="Cumulative frequency"),
    x=alt.X('static_wf', title="Static corpus frequency"),
    color="sample_name:N",
    opacity=alt.condition(selection_sampling, alt.value(1), alt.value(0.2)),
    tooltip=["sample_name", "word", "static_wf", "dynamic_wf"]
).add_selection(selection_epoch, selection_sampling).transform_filter(selection_epoch).interactive()

plot_ff.save(f'{working_directory}strain_static_dynamic_wf.html')
plot_ff



# %%
