# %% Environment
import os
import pandas as pd
import altair as alt
from IPython.display import clear_output
from importlib import reload
import data_wrangling, meta
reload(data_wrangling)

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

df = pd.read_csv()