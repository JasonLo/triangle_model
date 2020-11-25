import altair as alt
import meta
import data_wrangling
import pandas as pd
from IPython.display import clear_output

alt.data_transformers.disable_max_rows()
cfg = meta.model_cfg(json_file='models/booboo/model_config.json')
data = data_wrangling.MyData()


def dry_run_sampler(sample_name, cfg):
    cfg.sample_name = sample_name
    sampler = data_wrangling.Sampling(cfg, data)
    next(sampler.sample_generator())
    while sampler.debug_log_epoch[-1] <= 100:
        print(sampler.debug_log_epoch)
        next(sampler.sample_generator())
        clear_output(wait=True)

    df_corpus_size = pd.DataFrame({"epoch": sampler.debug_log_epoch,
                                   "corpus_size": sampler.debug_log_corpus_size})
    df_corpus_size.to_csv(
        f"working/dynamic_frequency/{sample_name}_corpus_size.csv")
    sampler.debug_log_dynamic_wf.to_csv(
        f"working/dynamic_frequency/{sample_name}_dynamic_corpus.csv")
    print("All done.")


# Dry run and log dynamic corpus
[dry_run_sampler(sample_name, cfg)
 for sample_name in ['experimental', 'chang', 'hs04', 'jay']]


# Compile corpus size
df1 = pd.read_csv('working/dynamic_frequency/chang_corpus_size.csv')
df1['sample_name'] = 'chang'
df2 = pd.read_csv('working/dynamic_frequency/experimental_corpus_size.csv')
df2['sample_name'] = 'continuous'
df3 = pd.read_csv('working/dynamic_frequency/hs04_corpus_size.csv')
df3['sample_name'] = 'hs04'
df4 = pd.read_csv('working/dynamic_frequency/jay_corpus_size.csv')
df4['sample_name'] = 'jay'
df = pd.concat([df1, df2, df3, df4], axis=0)
del df1, df2, df3, df4
df["epoch"] = df.epoch - 1

plot_dcs = alt.Chart(df).mark_line().encode(
    x="epoch:Q",
    y="corpus_size:Q",
    color="sample_name"
).properties(title="Dynamic corpus size").interactive()

plot_dcs

plot_dcs.save('working/dynamic_frequency/corpus_size.html')


# Compile dynamic corpus
df1 = pd.read_csv('working/dynamic_frequency/chang_dynamic_corpus.csv')
df1['sample_name'] = 'chang'
df2 = pd.read_csv('working/dynamic_frequency/experimental_dynamic_corpus.csv')
df2['sample_name'] = 'continuous'
df3 = pd.read_csv('working/dynamic_frequency/hs04_dynamic_corpus.csv')
df3['sample_name'] = 'hs04'
df4 = pd.read_csv('working/dynamic_frequency/jay_dynamic_corpus.csv')
df4['sample_name'] = 'jay'

df = pd.concat([df1, df2, df3, df4], axis=0)
del df1, df2, df3, df4

# Only select Strain items
sel_df = df.loc[df.word.isin(data.df_strain.word)]

sel_df = sel_df.melt(id_vars=['word', 'sample_name'],
             var_name='epoch_label', value_name='wf')
sel_df['epoch'] = sel_df.epoch_label.apply(lambda x: int(x[12::]) - 1)

sel_df.rename(columns={'wf': 'dynamic_wf'}, inplace=True)

# Down-sample epoch
sel_epoch = [0, 1, 3, 5, 7, 10, 30, 40, 50, 80, 100]
sel_df = sel_df.loc[sel_df.epoch.isin(sel_epoch),]

df_strain = pd.merge(sel_df, data.df_strain, "left", on='word')
df_strain['cond'] = df_strain.frequency + '_' + df_strain.pho_consistency
df_strain.rename(columns={'wf': 'static_wf'}, inplace=True)

# Dynamic frequncy in Strain by condition
plot_corpus_epoch = alt.Chart(df_strain).mark_line().encode(
    x="epoch:Q",
    y="mean(dynamic_wf):Q",
    color="sample_name:N",
    column="frequency:N",
    row="pho_consistency:N"
).interactive().properties(title="Dynamic frequency")

plot_corpus_epoch.save('working/dynamic_frequency/strain_dynamic_frequency.html')


# Relationship between actual corpus frequency and dynamic frequency
input_dropdown = alt.binding_select(options=sel_epoch)
selection_epoch = alt.selection_single(fields=['epoch'], bind=input_dropdown, init={'epoch': 100} )
selection_sampling = alt.selection_multi(fields=['sample_name'], bind='legend')

plot_ff = alt.Chart(df_strain).mark_point().encode(
    y=alt.Y('dynamic_wf', title="Dynamic frequency"),
    x=alt.X('static_wf', title="Static corpus frequency"),
    color="sample_name:N",
    opacity=alt.condition(selection_sampling, alt.value(1), alt.value(0.2)),
    tooltip=["sample_name", "word", "static_wf", "dynamic_wf"]
).add_selection(selection_epoch, selection_sampling).transform_filter(selection_epoch).interactive()

plot_ff.save('working/dynamic_frequency/strain_static_dynamic_wf.html')