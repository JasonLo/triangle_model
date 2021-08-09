import pickle
import pandas as pd
from matplotlib import pyplot as plt


class mn_weights:
    name_map = {
        "Phono -> psh": "w_hps_ph",
        "Con -> csh": "context_csh",
        "psh -> Semantics": "w_hps_hs",
        "csh -> Semantics": "context_sem",
        "Semantics -> SemCleanup": "w_sc",
        "SemCleanup -> Semantics": "w_cs",
        "Bias -> Semantics": "bias_s",
        "Bias -> SemCleanup": "bias_css",
        "Bias -> psh": "bias_hps",
        "Bias -> csh": "bias_hcs",
        "Semantics -> sph": "w_hsp_sh",
        "sph -> Phono": "w_hsp_hp",
        "Phono -> PhoCleanup": "w_pc",
        "PhoCleanup -> Phono": "w_cp",
        "Bias -> Phono": "bias_p",
        "Bias -> sph": "bias_hsp",
        "Bias -> PhoCleanup": "bias_cpp",
        "Ortho -> oph": "w_hop_oh",
        "Ortho -> osh": "w_hos_oh",
        "oph -> Phono": "w_hop_hp",
        "osh -> Semantics": "w_hos_hs",
        "Bias -> oph": "bias_hop",
        "Bias -> osh": "bias_hos",
    }

    def __init__(self, file: str = None, useful_start_idx: int = 25):
        self.weights = self.parse_weight(file)

        # Print keys
        self.useful_keys = list(self.weights.keys())[useful_start_idx:]
        self.useless_keys = list(self.weights.keys())[:useful_start_idx]
        print(f"Useful Keys: {self.useful_keys}\n")
        print(f"Useless Keys: {self.useless_keys}")

    @staticmethod
    def parse_weight(file: str = None) -> dict:
        """Parse mikenet weight file"""
        weight = {}
        key = None
        value = []

        if file is None:
            file = "Reading_Weight_v1"

        with open(file, "r") as f:
            for line in f:
                # Identify line is header (key) or value
                try:
                    line = float(line)
                except ValueError:
                    pass

                if type(line) is str:
                    # Write last record
                    if key is not None:
                        weight[key] = value

                    # Create new key and init value
                    key = line.strip()
                    value = []
                else:
                    value.append(line)

        # write last record
        weight[key] = value

        return weight

    def __repr__(self):
        return "\n".join([f"{i}: {x}" for i, x in enumerate(self.weights.keys())])

    def plot(self, weight_name: str, ax: plt.axes = None, xlim: tuple = None) -> plt.figure:
        """Density plot of a given weight"""
        df = pd.DataFrame({weight_name: self.weights[weight_name]})
        if len(df) > 1000:
            df = df.sample(1000)

        title = f"{weight_name}\n({self.as_tf_name(weight_name)})"
        return df.plot.density(title=title, ax=ax, legend=None, xlim=xlim)

    def plot_all(self, xlim: tuple = None) -> plt.figure:
        """Plot all the useful weights"""
        fig, ax = plt.subplots(5, 5, figsize=(25, 25), sharex=True)

        for i, weight_name in enumerate(self.useful_keys):
            self.plot(weight_name, ax=ax[i // 5, i % 5], xlim=xlim)

        return fig

    def as_tf_name(self, name):
        """Convert MikeNet weight name into TF weight name"""
        return self.name_map[name]

    def as_mn_name(self, name):
        """COnvert TF weight name into MN weight name"""
        reverse_map = {v: k for k, v in self.name_map.items()}
        return reverse_map[name]
