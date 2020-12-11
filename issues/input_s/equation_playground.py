# %% Equation playground
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm


sys.path.append("/home/jupyter/tf/src/")
import meta, data_wrangling

cfg = meta.ModelConfig.from_json(
    "../../models/test_sampling_speed_2/model_config.json")
data = data_wrangling.MyData()

# %%
from importlib import reload
reload(data_wrangling)

class SemanticExperiment:
    """Semantic experiment class for evaluating semantic equation"""
    # Mean in each condition in Strain data set (in a perfect world)
    strain = {
                "HF_HI": {"wf": 6500., "img": 6.},
                "HF_LI": {"wf": 6500., "img": 3.5},
                "LF_HI": {"wf": 400., "img": 6.},
                "LF_LI": {"wf": 400., "img": 3.5}
            }

    def __init__(self, **kwargs):
        self.semantic_params = kwargs
        self.sampler = self._run_sampling()
        self.df, self.df_mean = self._parse_sampler_results()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _run_sampling(self):
        last_epoch = 0
        sampler = data_wrangling.Sampling(cfg, data, debugging=True)
        sampler.set_semantic_parameters(**self.semantic_params)

        with tqdm(total=100) as progress:
            while sampler.current_epoch <= 100:

                # Progress bar
                if last_epoch != sampler.current_epoch:
                    progress.update(1)
                last_epoch = sampler.current_epoch

                # dry run sampling
                next(sampler.sample_generator(dryrun=True))

        return sampler

    def _parse_sampler_results(self):

        # Convert dict to df
        df_wf = pd.DataFrame(self.sampler.debug_wf).transpose()
        df_wf.columns = "cwf_" + df_wf.columns.astype(str)
        df_wf["word"] = df_wf.index

        df_sem = pd.DataFrame(self.sampler.debug_sem).transpose()
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

        # Activation
        df["activation"] = df.semantic_input.apply(self.sigmoid)

        # Mean df
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


        return (df, df_mean)

    def plot_input(self):

        plot_sem = alt.Chart(self.df).mark_line().encode(
            x="epoch:Q",
            y="mean(semantic_input):Q",
            color="cond:N",
        )

        plot_wf = alt.Chart(self.df).mark_line().encode(
            x="epoch:Q",
            y="mean(cumulative_frequency):Q",
            color="cond:N",
        )

        return plot_wf | plot_sem

    def plot_strain(self, df=None):

        if df is None:
            df = self.df_mean

        strain_conds = self.strain.keys()
        fig = plt.figure(figsize=(15, 10))

        # Semantic input
        ax = fig.add_subplot(221)
        ax.title.set_text("Semantic input over epoch")
        for condition in strain_conds:
            ax.plot(df.epoch, df["semantic_input"][condition], label=condition)
        ax.legend()

        # Semantic activation
        ax = fig.add_subplot(222)
        ax.title.set_text("Semantic activation over epoch")
        for condition in strain_conds:
            ax.plot(df.epoch, df["activation"][condition], label=condition)
        ax.legend()

        # Contrasts for input
        contrasts = ["frequency_effect", "imageability_effect", "fxi_interaction"]

        ax = fig.add_subplot(223)
        ax.title.set_text("Contrasts for input")
        for contrast in contrasts:
            ax.plot(df.epoch, df[f"{contrast}_semantic_input"], label=contrast)
        ax.legend()

        # Contrasts for activation
        ax = fig.add_subplot(224)
        ax.title.set_text("Contrasts for activation")

        for contrast in contrasts:
            ax.plot(df.epoch, df[f"{contrast}_activation"], label=contrast)
        ax.legend()


# %%
proto4 = SemanticExperiment(**{"g":100, "k":100, "d":100})
proto4.plot_strain()
