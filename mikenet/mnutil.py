import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


class MikeNetWeight:
    """A MikeNet Weight converter for grafting the weights from MikeNet to TensorFlow."""

    # A manual weight name mapping from MN to TF.
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
        self.weight_keys = list(self.weights.keys())[useful_start_idx:]
        self.nonweight_keys = list(self.weights.keys())[:useful_start_idx]
        print(f"Weight Keys: {self.weight_keys}\n")
        print(f"Non-weight Keys: {self.nonweight_keys}")

        # Get dimensions from biases
        self.sem_units = len(self.weights["Bias -> Semantics"])
        self.pho_units = len(self.weights["Bias -> Phono"])
        self.pho_cleanup_units = len(self.weights["Bias -> PhoCleanup"])
        self.sem_cleanup_units = len(self.weights["Bias -> SemCleanup"])
        self.hidden_os_units = len(self.weights["Bias -> osh"])
        self.hidden_op_units = len(self.weights["Bias -> oph"])
        self.hidden_ps_units = len(self.weights["Bias -> psh"])
        self.hidden_sp_units = len(self.weights["Bias -> sph"])

        self.ort_units = int(len(self.weights["Ortho -> oph"]) / self.hidden_op_units)

        self.shape_map = self.create_weights_shapes()
        self.weights_2d = self.reshape_all_weights()
        self.weights_tf = self.convert_all_weights_to_tf()

    @staticmethod
    def reshape_weight(weight, shape: tuple) -> np.array:
        """Reshape a weight into a matrix"""
        return np.array(weight).reshape(shape)

    @staticmethod
    def convert_to_tf_weights(weight2d, name) -> tf.Variable:
        """Convert a weight matrix into tensorflow format"""
        x = tf.Variable(weight2d, dtype=tf.float32, name=name)
        return x

    def create_weights_shapes(self):
        """Create a dictionary that consists of all the proper shape of each weights"""
        shape_map = {
            "Phono -> psh": (self.pho_units, self.hidden_ps_units),
            "psh -> Semantics": (self.hidden_ps_units, self.sem_units),
            "Semantics -> SemCleanup": (self.sem_units, self.sem_cleanup_units),
            "SemCleanup -> Semantics": (self.sem_cleanup_units, self.sem_units),
            "Semantics -> sph": (self.sem_units, self.hidden_sp_units),
            "sph -> Phono": (self.hidden_sp_units, self.pho_units),
            "Phono -> PhoCleanup": (self.pho_units, self.pho_cleanup_units),
            "PhoCleanup -> Phono": (self.pho_cleanup_units, self.pho_units),
            "Ortho -> oph": (self.ort_units, self.hidden_op_units),
            "Ortho -> osh": (self.ort_units, self.hidden_os_units),
            "oph -> Phono": (self.hidden_op_units, self.pho_units),
            "osh -> Semantics": (self.hidden_os_units, self.sem_units),
            "Bias -> oph": (self.hidden_op_units,),
            "Bias -> osh": (self.hidden_os_units,),
            "Bias -> Semantics": (self.sem_units,),
            "Bias -> Phono": (self.pho_units,),
            "Bias -> psh": (self.hidden_ps_units,),
            "Bias -> sph": (self.hidden_sp_units,),
            "Bias -> SemCleanup": (self.sem_cleanup_units,),
            "Bias -> PhoCleanup": (self.pho_cleanup_units,),
        }
        return shape_map

    def reshape_all_weights(self):
        """Reshape all the weights"""
        return {
            k: self.reshape_weight(self.weights[k], v)
            for k, v in self.shape_map.items()
        }

    def convert_all_weights_to_tf(self):
        """Convert all the weights to tensorflow format"""

        return {
            self.as_tf_name(k): self.convert_to_tf_weights(v, self.as_tf_name(k))
            for k, v in self.weights_2d.items()
        }

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

    def plot(
        self, weight_name: str, ax: plt.axes = None, xlim: tuple = None
    ) -> plt.figure:
        """Density plot of a given weight"""
        df = pd.DataFrame({weight_name: self.weights[weight_name]})
        if len(df) > 1000:
            df = df.sample(1000)

        tf_weight_name = self.as_tf_name(weight_name)
        title = f"{weight_name}\n({tf_weight_name})"
        color = "red" if tf_weight_name.startswith("bias") else "blue"
        return df.plot.density(title=title, ax=ax, legend=None, xlim=xlim, color=color)

    def plot_all(self, xlim: tuple = None) -> plt.figure:
        """Plot all the weights density"""
        fig, ax = plt.subplots(5, 5, figsize=(25, 25), sharex=True)

        for i, weight_name in enumerate(self.weight_keys):
            self.plot(weight_name, ax=ax[i // 5, i % 5], xlim=xlim)

        return fig

    def as_tf_name(self, name):
        """Convert MikeNet weight name into TF weight name"""
        return self.name_map[name]

    def as_mn_name(self, name):
        """COnvert TF weight name into MN weight name"""
        reverse_map = {v: k for k, v in self.name_map.items()}
        return reverse_map[name]

    def __repr__(self):
        return "\n".join([f"{i}: {x}" for i, x in enumerate(self.weights.keys())])
