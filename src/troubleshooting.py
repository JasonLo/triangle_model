import meta, data_wrangling, modeling, evaluate
import random, os
import pandas as pd
import altair as alt
import helper as H


class Diagnosis:
    """A diagnoistic bundle to trouble shot activation and input in semantic layer
    Usage:
    Step 1. Init by code_name 
    Step 2. Call eval() method to a) get the evaluation results from evaluate.TestSet object b) Load weight to the model at given epoch
    Step 3. Call set_target_word() method to "zoom in" a target word results (including all crucial input and activation pathways as defined in SEM_NAME_MAP and PHO_NAME_MAP)
    Step 4. plot_diagnosis
    """

    SEM_NAME_MAP = {
        "input_hps_hs": "PS",
        "input_css_cs": "CS",
        "input_sem_ss": "SS",
        "input_hos_hs": "OS",
        "input_sem": "input",
        "sem": "act",
    }
    PHO_NAME_MAP = {
        "input_hsp_hp": "SP",
        "input_cpp_cp": "CP",
        "input_pho_pp": "PP",
        "input_hop_hp": "OP",
        "input_pho": "input",
        "pho": "act",
    }

    def __init__(self, code_name: str):
        self.cfg = meta.ModelConfig.from_json(
            os.path.join("models", code_name, "model_config.json")
        )
        self.cfg.output_ticks = 13  # Full export

    def eval(self, testset_name: str, task: str, epoch: int):
        self.testset_package = data_wrangling.load_testset(
            os.path.join("dataset", "testsets", f"{testset_name}.pkl.gz")
        )

        self.model = modeling.MyModel(self.cfg)
        self.model.load_weights(self.cfg.saved_weights_fstring.format(epoch=epoch))
        self.model.set_active_task(task)
        input_name = modeling.IN_OUT[task][0]
        self.y_pred = self.model([self.testset_package[input_name]] * self.cfg.n_timesteps)

        # Get data for evaluate object
        self.testset = evaluate.TestSet(self.cfg)
        self.df = self.testset.eval(testset_name, task)
        self.df = self.df.loc[self.df.epoch == epoch]

    @property
    def list_outputs(self) -> list:
        return list(self.y_pred.keys())

    @property
    def list_weights(self) -> list:
        return [w.name for w in self.model.weights]

    @property
    def list_all_words(self) -> dict:
        words = self.testset_package["item"]
        return dict(zip(range(len(words)), words))

    @property
    def list_all_correct_words(self) -> dict:
        x = list(
            self.df.loc[
                (self.df.acc == 1) & (self.df.timetick == self.df.timetick.max()),
                "word",
            ].unique()
        )
        return {k: v for k, v in self.all_words.items() if v in x}

    @property
    def list_all_incorrect_words(self) -> dict:
        df = self.df
        df = df.loc[(df.acc == 0) & (df.timetick == df.timetick.max())]

        ic_pho_word = list(df.loc[df.output_name == "pho", "word"].unique())
        ic_sem_word = list(df.loc[df.output_name == "sem", "word"].unique())

        return {
            "pho": {k: v for k, v in self.all_words.items() if v in ic_pho_word},
            "sem": {k: v for k, v in self.all_words.items() if v in ic_sem_word},
        }

    def get_weight(self, name):
        """export the weight tensor"""
        return [
            w.numpy() for w in self.model.weights if w.name.endswith(f"{name}:0")
        ][0]

    def get_output(self, output_name, timetick=None):
        """Get selected output from TensorArrays"""
        if timetick is None:
            return self.y_pred[output_name][:, self.target_word_idx, :].numpy()
        else:
            return self.y_pred[output_name][timetick, self.target_word_idx, :].numpy()

    @property
    def list_output_phoneme(self) -> list:
        """Get output phoneme from model output activation pattern"""
        assert self.target_word is not None
        return H.get_batch_pronunciations_fast(self.get_output('pho'))

    def set_target_word(self, word: str) -> pd.DataFrame:
        self.target_word = word
        self.target_word_idx = self.testset_package["item"].index(self.target_word)
        self.word_sem_df = self.make_output_diagnostic_df(word, "sem")
        self.word_pho_df = self.make_output_diagnostic_df(word, "pho")

        print(f"Target pronounciation is: {self.testset_package['phoneme'][self.target_word_idx]}")
        return self.df.loc[(self.df.word==self.target_word) & (self.df.timetick == self.df.timetick.max())]

    def make_output_diagnostic_df(self, target_word: str, layer: str) -> pd.DataFrame:
        """Output all Semantic related input and activation in a word"""

        assert layer in ("pho", "sem")

        if layer == "pho":
            bias_name = "bias_p:0"
            name_map = self.PHO_NAME_MAP
        else:
            bias_name = "bias_s:0"
            name_map = self.SEM_NAME_MAP

        # Time invariant elements
        df_dict = {}
        df_dict["target_act"] = self.testset_package[layer][
            self.target_word_idx, :
        ].numpy()
        df_dict["bias"] = [
            w.numpy() for w in self.model.weights if w.name.endswith(bias_name)
        ][0]
        df_time_invar = pd.DataFrame.from_dict(df_dict)
        df_time_invar["unit"] = df_time_invar.index
        df_time_invar["word"] = target_word

        # Time varying elements
        df_time_varying = pd.DataFrame()

        for i, output_name in enumerate(name_map.keys()):
            this_output_df = pd.DataFrame()
            for t in range(13):
                df_dict = {}
                df_dict[name_map[output_name]] = self.y_pred[output_name][
                    t, self.target_word_idx, :
                ].numpy()
                this_step_df = pd.DataFrame.from_dict(df_dict)
                this_step_df["timetick"] = t
                this_step_df["unit"] = this_step_df.index
                this_output_df = pd.concat([this_output_df, this_step_df])

            if i == 0:
                df_time_varying = this_output_df
            else:
                df_time_varying = pd.merge(
                    df_time_varying, this_output_df, on=["timetick", "unit"]
                )

        # Merge and export
        df = df_time_varying.merge(df_time_invar, on="unit", how="left")
        df["unit_acc"] = abs(df.target_act - df.act) < 0.5
        df = df[
            ["word", "unit", "unit_acc", "timetick", "target_act", "bias"]
            + list(name_map.values())
        ]

        # Restructure
        melt_value_vars = ["bias"] + list(name_map.values())
        return df.melt(
            id_vars=["word", "unit", "timetick", "target_act", "unit_acc"],
            value_vars=melt_value_vars,
        )

    @staticmethod
    def find_incorrect_nodes(df: pd.DataFrame, layer: str) -> pd.DataFrame:
        """Based on last time tick, create a dataframe with all incorrect nodes in a given output layer"""
        assert layer in ("pho", "sem")
        return df.loc[(df.unit_acc == 0) & (df.timetick == df.timetick.max())]

    @staticmethod
    def find_correct_nodes(df: pd.DataFrame, layer: str) -> pd.DataFrame:
        """Based on last time tick, create a dataframe with all correct nodes in a given output layer"""
        assert layer in ("pho", "sem")
        return df.loc[(df.unit_acc == 1) & (df.timetick == df.timetick.max())]

    def subset_df(self, layer: str, target_act: int) -> pd.DataFrame:
        """Subset the dataset to at most 10 correct + 10 incorrect nodes"""
        assert layer in ("pho", "sem")
        df = self.word_pho_df if layer == "pho" else self.word_sem_df
        df = df.loc[df.target_act == target_act]

        # Get at most 10 correct units
        correct_units = list(self.find_correct_nodes(df, layer).unit.unique())
        total_correct_units = len(correct_units)
        correct_units = (
            random.sample(correct_units, 10)
            if total_correct_units > 10
            else correct_units
        )

        # Get at most 10 incorrect units
        incorrect_units = list(self.find_incorrect_nodes(df, layer).unit.unique())
        total_incorrect_units = len(incorrect_units)
        incorrect_units = (
            random.sample(incorrect_units, 10)
            if total_incorrect_units > 10
            else incorrect_units
        )

        print(f"At last timetick in {layer} output layer (target = {target_act}):")
        print(f"Selected {len(correct_units)} out of {total_correct_units} correct units")
        print(f"Selected {len(incorrect_units)} out of {total_incorrect_units} incorrect units")

        return df.loc[df.unit.isin(correct_units + incorrect_units)]

    def plot_one_node(self, layer:str, node:int) -> alt.Chart:
        assert layer in ("pho", "sem")
        df = self.word_pho_df if layer == "pho" else self.word_sem_df
        df = df.loc[df.unit == node]
        p = Plots(df)
        return p.raw_and_tai()

    def plot_one_layer_by_target(self, layer: str, target_act: int) -> alt.Chart:
        df = self.subset_df(layer, target_act)
        p = Plots(df)
        return p.raw_and_tai().properties(title=f"At target node = {target_act}")

    def plot_one_layer(self, layer: str) -> alt.Chart:
        p1 = self.plot_one_layer_by_target(layer, target_act=1)
        p2 = self.plot_one_layer_by_target(layer, target_act=0)
        return (p1 & p2).resolve_scale(y="shared").properties(title=f"In a word: {self.target_word}")


class Plots:
    def __init__(self, df):
        self.df = df
        self.sel_var = alt.selection_multi(fields=["variable"], bind="legend")
        self.sel_unit = alt.selection_multi(fields=["unit"])

    def raw_inputs(self) -> alt.Chart():
        """Plot raw input by pathway without sel unit dependency"""
        return (
            alt.Chart(self.df.loc[~self.df.variable.isin(["act", "input"])])
            .mark_line()
            .encode(
                y="mean(value):Q",
                x="timetick",
                color="variable",
                opacity=alt.condition(self.sel_var, alt.value(1), alt.value(0.2)),
            )
            .add_selection(self.sel_var)
        ).properties(title=f"Time course of raw input from each pathway")

    def _subplot_input_pathways(self) -> alt.Chart:
        """Plot input by pathway"""
        return (
            alt.Chart(self.df.loc[~self.df.variable.isin(["act", "input"])])
            .mark_line()
            .encode(
                y="mean(value):Q",
                x="timetick",
                color="variable",
                opacity=alt.condition(self.sel_var, alt.value(1), alt.value(0.2)),
            )
            .add_selection(self.sel_var)
            .transform_filter(self.sel_unit)
        ).properties(title=f"Time course of raw input from each pathway")

    def _subplot_input_units(self) -> alt.Chart:
        """Plot time averaged input by unit"""
        line = (
            alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y")
        )
        return (
            line
            + (
                alt.Chart(self.df.loc[self.df.variable == "input"])
                .mark_line(point=True)
                .encode(
                    y="value:Q",
                    x="timetick",
                    color="unit:N",
                    opacity=alt.condition(self.sel_unit, alt.value(1), alt.value(0.2)),
                    tooltip=["unit", "value"],
                )
                .add_selection(self.sel_unit)
            )
        ).properties(title=f"Time course of time averaged input in each unit")

    def _subplot_act(self) -> alt.Chart:
        """Plot activation by accuracy"""
        color_scale = alt.Scale(domain=[False, True], range=["#c30d24", "#1770ab"])
        line = (
            alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color="black").encode(y="y")
        )
        return line + (
            alt.Chart(self.df.loc[self.df.variable == "act"])
            .mark_circle()
            .encode(
                y=alt.Y("value:Q", scale=alt.Scale(domain=(0, 1))),
                x="timetick",
                detail="unit:N",
                color=alt.Color("unit_acc:N", scale=color_scale),
                tooltip=["unit"],
            )
            .transform_filter(self.sel_unit)
            .add_selection(self.sel_unit)
            .properties(title=f"Activation time course in each unit")
        )

    def raw_and_tai(self):
        return (
            self._subplot_input_pathways() | self._subplot_input_units()
        ).resolve_scale(color="independent", y="shared") 