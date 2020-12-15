# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SemanticExperiment:
    """Semantic experiment class for evaluating semantic equation"""

    # Mean in each condition in Strain data set
    strain = {
                "hi_freq_hi_img": {"wf": 6443.925, "img": 6.009909},
                "hi_freq_lo_img": {"wf": 8956.975, "img": 3.582470},
                "lo_freq_hi_img": {"wf": 293.475, "img": 6.252500},
                "lo_freq_lo_img": {"wf": 510.225, "img": 3.657500}
            }

    def __init__(self, g, k):
        self.g = g
        self.k = k
        self.df = self._make_df()

    def semantic_input(self, T, f):
        """Semantic equation
        T: epoch
        f: word frequency
        """
        numer = self.g * np.log(f + 2) * T
        denom = np.log(f + 2) * T + self.k
        return numer / denom



    def _make_df(self):
        df = pd.DataFrame()
        df["epoch"] = np.arange(0, self.k)

        for condition in self.strain.keys():
            df[f"{condition}_input"] = df.epoch.apply(
                self.semantic_input, f=self.strain[condition]["wf"])
            df[f"{condition}_activation"] = df[f"{condition}_input"].apply(lambda x: 1/(1 + np.exp(-x)))

        # Contrasts
        for measure in ["input", "activation"]:
            df[f"frequency_effect_{measure}"] = df[f"hi_freq_hi_img_{measure}"] + \
                                                     df[f"hi_freq_lo_img_{measure}"] - \
                                                     df[f"lo_freq_hi_img_{measure}"] - \
                                                     df[f"lo_freq_lo_img_{measure}"] 

            df[f"imageability_effect_{measure}"] = df[f"hi_freq_hi_img_{measure}"] + \
                                                     df[f"lo_freq_hi_img_{measure}"] - \
                                                     df[f"hi_freq_lo_img_{measure}"] - \
                                                     df[f"lo_freq_lo_img_{measure}"] 

            df[f"fxi_interaction_{measure}"] = df[f"lo_freq_hi_img_{measure}"] - \
                                                     df[f"lo_freq_lo_img_{measure}"] - \
                                                     df[f"hi_freq_hi_img_{measure}"] + \
                                                     df[f"hi_freq_lo_img_{measure}"] 



        return df

    def plot(self):

        fig = plt.figure(figsize=(15, 10))

        # Semantic input
        ax = fig.add_subplot(221)
        ax.title.set_text("Semantic input over epoch")
        for condition in self.strain.keys():
            ax.plot(self.df.epoch, self.df[f"{condition}_input"], label=condition)
        ax.legend()

        # Semantic activation
        ax = fig.add_subplot(222)
        ax.title.set_text("Semantic activation over epoch")
        for condition in self.strain.keys():
            ax.plot(self.df.epoch, self.df[f"{condition}_activation"], label=condition)
        ax.legend()

        # Contrasts for input
        contrasts = ["frequency_effect", "imageability_effect", "fxi_interaction"]
        
        ax = fig.add_subplot(223)
        ax.title.set_text("Contrasts for input")
        for contrast in contrasts:
            ax.plot(self.df.epoch, self.df[f"{contrast}_input"], label=contrast)
        ax.legend()

        # Contrasts for activation
        ax = fig.add_subplot(224)
        ax.title.set_text("Contrasts for activation")

        for contrast in contrasts:
            ax.plot(self.df.epoch, self.df[f"{contrast}_activation"], label=contrast)
        ax.legend()


# %%
SemanticExperiment(g=5, k=100).plot()   

# %%
