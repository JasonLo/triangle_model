# %% Environment
import os
import pandas as pd
import altair as alt
from IPython.display import clear_output
from scipy.stats import pearsonr
import data_wrangling, meta
alt.data_transformers.disable_max_rows()

os.chdir("/home/jupyter/tf")
cfg = meta.ModelConfig.from_json('models/boo/model_config.json')
cfg.sampling_plateau = 500,000
data = data_wrangling.MyData()
working_directory = "issues/developmental_sampler/"

# %% function to run sampling  

def dry_run_sampler(sample_name, file_name, cfg, output_folder):
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
        f"{output_folder}{file_name}_corpus_size.csv")
    
    dynamic_frequency_df = pd.concat(
        [pd.DataFrame.from_dict(x, orient='index', columns=[f"epoch_{i}"]) 
        for i, x in enumerate(sampler.debug_wf)], axis=1
        )

    dynamic_frequency_df.to_csv(
        f"{output_folder}{file_name}_dynamic_corpus.csv")

    print("All done.")


#%% Brute-force sampler

def run_and_plot(wf_low_clip, wf_high_clip, wf_compression, sampling_plateau):
    """ Simulate sampling stragegy"""
    

    code_name = f"r_lc{wf_low_clip}_hc{wf_high_clip}_com{wf_compression}_plateau{sampling_plateau}"

    def load_data_from_csv(implementations):
        df = pd.DataFrame()
        df_corpus_size = pd.DataFrame()

        for i in implementations:
            tmp_df_dwf = pd.read_csv(f"{working_directory}{i}_dynamic_corpus.csv")
            tmp_df_dwf['sample_name'] = i
            df = pd.concat([df, tmp_df_dwf], axis=0)

            tmp_df_corpus_size = pd.read_csv(f"{working_directory}{i}_corpus_size.csv")
            tmp_df_corpus_size['sample_name'] = i
            df_corpus_size = pd.concat([df_corpus_size, tmp_df_corpus_size], axis=0)

        return df, df_corpus_size

    try:
        # Try to read sampling dry run results from disk
        df, df_corpus_size = load_data_from_csv(["chang_jml", code_name])
    except:
        # Or dry run from scratch
        cfg.sample_name = "flexi_rank"
        cfg.wf_low_clip = wf_low_clip
        cfg.wf_high_clip = wf_high_clip
        cfg.wf_compression = wf_compression
        cfg.sampling_plateau = sampling_plateau
        dry_run_sampler(cfg.sample_name, code_name, cfg, working_directory)

        df, df_corpus_size = load_data_from_csv(["chang_jml", code_name])

    # Tidying
    df = df.rename({'Unnamed: 0': 'word'}, axis=1)
    df = df.melt(id_vars=['word', 'sample_name'], var_name='epoch_label', value_name='dwf')
    df['epoch'] = df.epoch_label.apply(lambda x: int(x[6::]))


    # Merge WSJ frequnecy and save
    df_train = pd.read_csv('dataset/df_train.csv')[['word', 'wf']]
    df = df.merge(df_train)

    df['bin'] = pd.qcut(df.wf, q=5, labels=['lowest', 'low', 'mid', 'high', 'highest'])
    df = df.groupby(['epoch', 'sample_name', 'bin']).mean().reset_index()

    # Plotting
    p = alt.Chart(df).encode(
        x="epoch", 
        y="mean(dwf)",
        color=alt.Color("bin", sort=["lowest", "low", "mid", "high", "highest"]),
        column="sample_name:N",
        tooltip=['wf', 'dwf']
    ).mark_line().interactive()
    
    p2 = alt.Chart(df_corpus_size).encode(x="epoch", y="corpus_size", column="sample_name:N").mark_line()
    return p & p2

#%%
eq0 = run_and_plot(wf_low_clip=0, wf_high_clip=30000, wf_compression="root", sampling_plateau=500_000)
eq2 = run_and_plot(wf_low_clip=0, wf_high_clip=30000, wf_compression="log", sampling_plateau=500_000)

#%% Failed trash

eq1 = run_and_plot(wf_low_clip=0, wf_high_clip=30000, wf_compression="log", sampling_speed=4.)
eq3 = run_and_plot(wf_low_clip=0, wf_high_clip=30000, wf_compression="root", sampling_speed=4.)
eq4 = run_and_plot(wf_low_clip=0, wf_high_clip=10000, wf_compression="root", sampling_speed=2.)
eq5 = run_and_plot(wf_low_clip=0, wf_high_clip=3000, wf_compression="root", sampling_speed=2.)


#%% Sponteneous sampling distribution



# Load dry run dynamic corpus
def load_dynamic_corpus(name):

    df = pd.read_csv(os.path.join(working_directory, f'{name}_dynamic_corpus.csv'))
    df.rename(columns={'Unnamed: 0': 'word'}, inplace=True)

    tmp = data.df_train[['word', 'wf', 'gr14']].copy()
    tmp['rank_wf_wsj'] = tmp.wf.rank(ascending=False)
    tmp['rank_wf_zeno'] = tmp.gr14.rank(ascending=False)

# Calculate sponteneous freuqncy from cumulative frequency 

    for x in range(1, 101):
        df[f'delta_{x}'] = df[f'epoch_{x}'] - df[f'epoch_{x-1}']
        try:
            df[f'delta_{x}_bin'] = pd.qcut(df[f'delta_{x}'], q=5, labels=['lowest', 'low', 'mid', 'high', 'highest'])
        except:
            pass

    return df.merge(tmp, how='left', on='word')

chang_df = load_dynamic_corpus('chang_jml')
my_df = load_dynamic_corpus('r_lc0_hc30000_comlog_plateau500000')




# %% This is what Jay wants
def plot_word_rank_density(df, epoch, x='rank_wf_wsj'):
    v = f'delta_{epoch}'
    plot = alt.Chart(df.loc[(df[v]>0) & (df[x] <= 2000)]).mark_bar(
    ).encode(
        x=alt.X(f'{x}:Q', scale=alt.Scale(domain=(0, 2000)), bin=alt.Bin(extent=[0, 2000], step=200)),
        y=alt.Y(f'sum({v})', scale=alt.Scale(domain=(0, 10000))),
    ).properties(width=100, height=100, title=f'epoch={epoch}')

    return plot


#%%
chang_on_zeno_plot = alt.hconcat()
chang_on_wsj_plot = alt.hconcat()
ours_on_wsj_plot = alt.hconcat()
ours_on_zeno_plot = alt.hconcat()

for epoch in range(1, 9):
    chang_on_zeno_plot |= plot_word_rank_density(chang_df, epoch=epoch, x='rank_wf_zeno')
    ours_on_zeno_plot |= plot_word_rank_density(my_df, epoch=epoch, x='rank_wf_zeno')
    chang_on_wsj_plot |= plot_word_rank_density(chang_df, epoch=epoch, x='rank_wf_wsj')
    ours_on_wsj_plot |= plot_word_rank_density(my_df, epoch=epoch, x='rank_wf_wsj')


combine_plot = (
    chang_on_zeno_plot.properties(title='Chang on Zeno frequency ordering') &
    ours_on_zeno_plot.properties(title='Ours on Zeno frequency ordering') &
    chang_on_wsj_plot.properties(title='Chang on WSJ frequency ordering') &
    ours_on_wsj_plot.properties(title='Ours on WSJ frequency ordering')
)

combine_plot.save('histogram_compare.html')


#%% This is not what Jay's want

def plot_sponteneous_density(df, epoch=1):
    v = f'delta_{epoch}'
    plot = alt.Chart(df.loc[df[v]>0]).mark_area(
    ).transform_density(v, as_=['sponteneous_f', 'density']
    ).encode(
        x=alt.X('sponteneous_f:Q',scale=alt.Scale(domain=(0,200))),
        y=alt.Y('density:Q', scale=alt.Scale(domain=(0, 0.1))),
    ).properties(width=100, height=100, title=f'epoch={epoch}')

    return plot

def plot_10(df):
    plot=alt.hconcat()
    for epoch in range(1,11):
        plot |= plot_sponteneous_density(df, epoch)
    return plot

# %% 

plot_sponteneous_density(chang_df, epoch =1 )

#%%

chang = plot_10(chang_df)
mine = plot_10(my_df)
#%%
comparison = (chang & mine)
comparison.save(os.path.join(working_directory, 'density_comparison.html'))


# %% Correlation of sampling probability

rs = [pearsonr(my_df[f'delta_{x}'], chang_df[f'delta_{x}'])[0] for x in range(1, 101)]

# %%
epoch = list(range(1, 101))
r_plot = alt.Chart(pd.DataFrame({'rs':rs, 'epoch':epoch})).mark_line().encode(
    y='rs:Q', x='epoch'
    ).properties(title='Pearson correlation between sponteneous frequency in Chang 2019 and our implementation')

r_plot.save(os.path.join(working_directory, 'correlation_of_sponteneous_f.html'))


#%% Point ot point
rs_delta = [pearsonr(chang_df[f'delta_{x}'], chang_df[f'delta_{x+1}'])[0] for x in range(1, 100)]

alt.Chart(pd.DataFrame({'rs_delta':rs_delta, 'epoch':list(range(1, 100))})).mark_line().encode(
    y='rs_delta:Q', x='epoch'
)
# %%
