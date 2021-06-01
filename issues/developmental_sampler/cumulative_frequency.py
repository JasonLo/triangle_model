# %% Environment
import os
import pandas as pd
import altair as alt
from IPython.display import clear_output
import data_wrangling, meta

os.chdir("/home/jupyter/tf")
cfg = meta.ModelConfig.from_json('models/boo/model_config.json')
cfg.sampling_speed = 2
data = data_wrangling.MyData()
working_directory = "issues/developmental_sampler/"

# %% function to run sampling  

def dry_run_sampler(sample_name, cfg, output_folder=working_directory):
    # Instantiate
    cfg.sample_name = sample_name
    sampler = data_wrangling.Sampling(cfg, data, debugging=True)

    # Dry run sampler to get dynamic corpus record
    while sampler.current_epoch <= 100:
        print(f"Running {sample_name} epoch:{sampler.current_epoch}")
        next(sampler.sample_generator(x="ort", y="pho", dryrun=True))
        clear_output(wait=True)

    df_corpus_size = pd.DataFrame({"epoch": sampler.debug_epoch,
                                   "corpus_size": sampler.debug_corpus_size})
    df_corpus_size.to_csv(
        f"{output_folder}{sample_name}_corpus_size.csv")
    
    dynamic_frequency_df = pd.concat(
        [pd.DataFrame.from_dict(x, orient='index', columns=[f"epoch_{i}"]) 
        for i, x in enumerate(sampler.debug_wf)], axis=1
        )

    dynamic_frequency_df.to_csv(
        f"{output_folder}{sample_name}_dynamic_corpus.csv")

    print("All done.")


# %% Do the works on multiple sampling implementations
implementations = ["chang_jml", "hs04", "jay", "developmental_rank_frequency"]
[dry_run_sampler(x, cfg) for x in implementations]

# %% Combine data
df = pd.DataFrame()
for i in implementations:
    tmp_df = pd.read_csv(f"{working_directory}{i}_dynamic_corpus.csv")
    tmp_df['sample_name'] = i
    df = pd.concat([df, tmp_df], axis=0)

# Tidying
df = df.rename({'Unnamed: 0': 'word'}, axis=1)
df = df.melt(id_vars=['word', 'sample_name'], var_name='epoch_label', value_name='dwf')
df['epoch'] = df.epoch_label.apply(lambda x: int(x[6::]))

# Down-sample epoch
# sel_epoch = [0, 1, 3, 5, 7, 10, 30, 40, 50, 80, 100]
# df = df.loc[df.epoch.isin(sel_epoch), ]

# Merge WSJ frequnecy and save
df_train = pd.read_csv('dataset/df_train.csv')[['word', 'wf']]
df = df.merge(df_train)




#%%

df['bin'] = pd.qcut(df.wf, q=5, labels=['lowest', 'low', 'mid', 'high', 'highest'])
df = df.groupby(['epoch', 'sample_name', 'bin']).mean().reset_index()




#%%
p = alt.Chart(df).encode(
    x="epoch", 
    y="mean(dwf)",
    color='sample_name:N',
    column=alt.Column("bin",sort=["lowest", "low", "mid", "high", "highest"]),
    tooltip=['wf', 'dwf']
).mark_line().interactive()
  

p.save(os.path.join(working_directory, "q5_compare_sample.html"))

# %%
