""" This module is for all things related to testsets
"""

from tqdm import tqdm
import os
import tensorflow as tf
import metrics, modeling, gcp
from data_wrangling import load_testset
from helper import get_batch_pronunciations_fast
import pandas as pd
import numpy as np


class TestSet:
    """Universal test set object for evaluating model results
    1. Single condition, single metric, single value output for maximum compatibility
    2. Model level info should be stored at separate table, and merge it in the end
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None  # Will create in eval()
        self.ckpt = None  # Will create in eval()
        self.metrics = {
            "acc": {"pho": metrics.PhoAccuracy(), "sem": metrics.RightSideAccuracy()},
            "sse": metrics.SumSquaredError(),
            "act0": metrics.OutputOfZeroTarget(),
            "act1": metrics.OutputOfOneTarget(),
        }

    def eval_train(self, task: str, n: int = 12, to_bq: bool = False):
        """Evaluate the full training set with batching.
        Due to memory demands, will not store df in memory, but save to csv file
        """

        for i in range(n):
            self.eval(f"train_batch_{i}", task, to_bq=to_bq)

    def eval(
        self,
        testset_name: str,
        task: str,
        save_file_prefix: str = None,
        to_bq: bool = False,
    ):
        """
        Inputs
        testset_name: name of testset, must match testset package (*.pkl.gz) name
        task: 1 of 9 task option in triangle model
        output: pandas dataframe with all the evaluation results
        """
        try:
            df = self.load(testset_name, task, save_file_prefix)
            print(f"Eval results found, load from saved csv")
        except (FileNotFoundError, IOError):

            df = pd.DataFrame()
            ts_path = "dataset/testsets"
            testset_package = load_testset(
                os.path.join(ts_path, f"{testset_name}.pkl.gz")
            )

            # Enforceing batch_size dim to match with test case
            inputs = testset_package[modeling.IN_OUT[task][0]]

            # Build model and switch task
            self.model = modeling.MyModel(self.cfg, batch_size_override=inputs.shape[0])
            self.ckpt = tf.train.Checkpoint(model=self.model)
            self.model.set_active_task(task)

            for epoch in tqdm(
                self.cfg.saved_epochs, desc=f"Evaluating {testset_name}:{task}"
            ):
                # for epoch in tqdm(range(250, 291, 10)):
                saved_checkpoint = self.cfg.saved_checkpoints_fstring.format(
                    epoch=epoch
                )
                self.ckpt.restore(
                    saved_checkpoint
                ).expect_partial()  # Only load weights
                y_pred = self.model([inputs] * self.cfg.n_timesteps)

                for timetick_idx in range(self.cfg.output_ticks):
                    if task == "triangle":
                        for output_name in ("pho", "sem"):
                            df = self._try_to_run_eval(
                                df,
                                y_pred,
                                testset_name,
                                task,
                                epoch,
                                output_name,
                                timetick_idx,
                                testset_package,
                            )
                    else:
                        output_name = modeling.IN_OUT[task][1]
                        df = self._try_to_run_eval(
                            df,
                            y_pred,
                            testset_name,
                            task,
                            epoch,
                            output_name,
                            timetick_idx,
                            testset_package,
                        )

            # Save evaluation
            if save_file_prefix is not None:
                csv_name = os.path.join(
                    self.cfg.eval_folder,
                    f"{save_file_prefix}_{testset_name}_{task}.csv",
                )
            else:
                csv_name = os.path.join(
                    self.cfg.eval_folder, f"{testset_name}_{task}.csv"
                )

            df.to_csv(csv_name)

            if to_bq:
                gcp.df_to_bigquery(df, self.cfg.batch_name, "train")
        return df

    def _try_to_run_eval(
        self,
        df,
        y_pred,
        testset_name,
        task,
        epoch,
        output_name,
        timetick_idx,
        testset_package,
    ):

        if testset_package[output_name] is not None:

            tag = {
                "code_name": self.cfg.code_name,
                "epoch": epoch,
                "testset": testset_name,
                "task": task,
                "output_name": output_name,
                "timetick_idx": timetick_idx,
                "timetick": self.output_idx_to_timetick(timetick_idx),
                "word": testset_package["item"],
                "cond": testset_package["cond"],
            }

            df = df.append(
                self._eval_one(y_pred, testset_package, tag), ignore_index=True
            )

        return df

    def load(self, testset_name, task, save_file_prefix=None):
        if save_file_prefix is not None:
            csv_file = os.path.join(
                self.cfg.eval_folder, f"{save_file_prefix}_{testset_name}_{task}.csv"
            )
        else:
            csv_file = os.path.join(self.cfg.eval_folder, f"{testset_name}_{task}.csv")
        return pd.read_csv(csv_file, index_col=0)

    def output_idx_to_timetick(self, idx):
        # Zero indexing idx to one indexing step
        d = self.cfg.n_timesteps - self.cfg.output_ticks
        return idx + 1 + d

    def _eval_one(self, y_pred, y_true, tag):
        """
        y_pred: predition dictionary, e.g., {'pho': (time ticks, items, output nodes)}
        y_true: label dictionary (time invarying), e.g., {'sem': (items, maybe n ans. output nodes)}
        """
        out = pd.DataFrame()

        this_y_pred = y_pred[tag["output_name"]][tag["timetick_idx"]]
        # shape: (time ticks, items, output nodes)

        this_y_true = y_true[tag["output_name"]]

        try:
            if tag["output_name"] == "pho":
                this_y_true_phoneme = y_true["phoneme"]
        except:
            print("Cannot find phoneme in y_true dictionary")
        # shape: (item, *maybe n ans, output nodes)

        acc = self.metrics["acc"][tag["output_name"]]
        sse = self.metrics["sse"]
        act0 = self.metrics["act0"]
        act1 = self.metrics["act1"]

        if type(this_y_true) is list:
            # List mode (for Glushko)
            out["acc"] = acc.item_metric_multi_list(this_y_true_phoneme, this_y_pred)
            out["sse"] = sse.item_metric_multi_list(this_y_true, this_y_pred)
            # TODO: add act0 and act1
        elif tf.rank(this_y_true) == 3:
            # Multi ans mode if we have 3 dims
            out["acc"] = acc.item_metric_multi_ans(this_y_true, this_y_pred)
            out["sse"] = sse.item_metric_multi_ans(this_y_true, this_y_pred)
            # TODO: add act0 and act1
        else:
            # Single ans mode
            out["acc"] = acc.item_metric(this_y_true, this_y_pred)
            out["sse"] = sse.item_metric(this_y_true, this_y_pred)
            out["act0"] = act0.item_metric(this_y_true, this_y_pred)
            out["act1"] = act1.item_metric(this_y_true, this_y_pred)

        # Write prediction if output is pho
        if tag["output_name"] == "pho":
            out["pho_pred"] = get_batch_pronunciations_fast(this_y_pred)

        # Write tag to df
        for k, v in tag.items():
            out[k] = v

        return out


# class Eval:
#     TESTSETS_NAME = ("strain", "grain")

#     def __init__(self, cfg, model, data):
#         self.cfg = cfg
#         self.data = data
#         self.model = model

#     def _load_results_from_file(self):
#         for testset_name in self.TESTSETS_NAME:
#             with os.path.join(self.cfg.eval_folder, f"{testset_name}_mean_df.csv") as f:
#                 try:
#                     setattr(self, f"{testset_name}_mean_df", pd.read_csv(f))
#                 except (FileNotFoundError, IOError):
#                     pass

#     def _eval_tasks(self, task, testset_name):
#         """The oral evaluations consists of multiple tasks, sp, ps, pp, ss
#         This function will:
#         1. Create the four tasks (TestSet object) based on testset_name
#         2. Evaluate the tasks
#         3. Concatenate all results into output df
#         """
#         df = pd.DataFrame()
#         x, y = modeling.IN_OUT[task]
#         this_test = TestSet(
#             name=testset_name,
#             cfg=self.cfg,
#             model=self.model,
#             task=task,
#             testitems=self.data.testsets[testset_name]["item"],
#             x_test=self.data.testsets[testset_name][x],
#             y_test=self.data.testsets[testset_name][y],
#         )

#         this_test.eval_all()
#         df = df.append(this_test.result, ignore_index=True)

#         return df


# class EvalOral:
#     """Bundle of testsets for Oral stage
#     Not finished... only have train strain and taraban cortese img
#     """

#     TESTSETS_NAME = ("train", "strain", "taraban")

#     def __init__(self, cfg, model, data):
#         self.cfg = cfg
#         self.model = model
#         self.data = data

#         self.train_mean_df = None
#         self.strain_mean_df = None
#         self.grain_mean_df = None
#         self.taraban_mean_df = None
#         self.cortese_mean_df = None
#         self.cortese_img_mean_df = None
#         self.homophone_mean_df = None

#         # Load eval results from file
#         for _testset_name in self.TESTSETS_NAME:
#             try:
#                 _file = os.path.join(
#                     self.cfg.model_folder,
#                     "eval",
#                     f"{_testset_name}_mean_df.csv",
#                 )
#                 setattr(self, f"{_testset_name}_mean_df", pd.read_csv(_file))
#             except (FileNotFoundError, IOError):
#                 pass

#         # Bundle testsets into dictionary
#         self.run_eval = {
#             "train": self._eval_train,
#             "strain": self._eval_strain,
#             "taraban": self._eval_taraban,
#             "cortese_img": self._eval_img,
#             "homophone": self._eval_homophone,
#         }

#     def eval(self, testset_name):
#         """Run eval and push to dat"""
#         if getattr(self, f"{testset_name}_mean_df") is None:
#             results = self.run_eval[testset_name]()
#         else:
#             print("Evaluation results found, loaded from file.")

#     def _eval_oral_tasks(self, testset_name):
#         """The oral evalution consists of multiple tasks, sp, ps, pp, ss
#         This function will:
#         1. Create the four tasks (TestSet object) based on testset_name
#         2. Evaluate the tasks
#         3. Concatenate all results into output df
#         """

#         df = pd.DataFrame()

#         tasks = ("pho_sem", "sem_pho", "sem_sem", "pho_pho")

#         for this_task in tasks:

#             x, y = this_task.split("_")
#             this_testset_object = TestSet(
#                 name=testset_name,
#                 cfg=self.cfg,
#                 model=self.model,
#                 task=this_task,
#                 testitems=self.data.testsets[testset_name]["item"],
#                 x_test=self.data.testsets[testset_name][x],
#                 y_test=self.data.testsets[testset_name][y],
#             )

#             this_testset_object.eval_all()
#             df = df.append(this_testset_object.result, ignore_index=True)

#         return df

#     # Different _eval_xxx bundle has slightly differnt post-processing needs,
#     # separate into multiple functions
#     def _eval_train(self):
#         testset_name = "train"
#         df = self._eval_oral_tasks(testset_name)

#         # Write item level results
#         df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", f"{testset_name}_item_df.csv")
#         )

#         # Aggregate
#         mean_df = (
#             df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
#             .mean()
#             .reset_index()
#         )
#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", f"{testset_name}_mean_df.csv")
#         )

#         self.train_mean_df = mean_df

#         return df

#     def _eval_homophone(self):
#         df = pd.DataFrame()
#         testsets = ("non_homophone", "homophone")

#         for testset_name in testsets:
#             df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "homophone_item_df.csv"))

#         mean_df = (
#             df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
#             .mean()
#             .reset_index()
#         )

#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", "homophone_mean_df.csv")
#         )

#         return df

#     def _eval_img(self):
#         df = pd.DataFrame()
#         testsets = (
#             "cortese_3gp_high_img",
#             "cortese_3gp_med_img",
#             "cortese_3gp_low_img",
#         )

#         for testset_name in testsets:
#             df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "img_item_df.csv"))

#         return df

#     def _eval_strain(self):
#         df = pd.DataFrame()
#         testsets = (
#             "strain_hf_con_hi",
#             "strain_hf_inc_hi",
#             "strain_hf_con_li",
#             "strain_hf_inc_li",
#             "strain_lf_con_hi",
#             "strain_lf_inc_hi",
#             "strain_lf_con_li",
#             "strain_lf_inc_li",
#         )

#         for testset_name in testsets:
#             df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "strain_item_df.csv"))

#         # Condition level aggregate
#         mean_df = (
#             df.groupby(
#                 [
#                     "code_name",
#                     "task",
#                     "testset",
#                     "epoch",
#                     "timetick",
#                     "y",
#                 ]
#             )
#             .mean()
#             .reset_index()
#         )
#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", "strain_mean_df.csv")
#         )
#         self.strain_mean_df = mean_df

#         return df

#     def _eval_taraban(self):

#         testsets = (
#             "taraban_hf-exc",
#             "taraban_hf-reg-inc",
#             "taraban_lf-exc",
#             "taraban_lf-reg-inc",
#             "taraban_ctrl-hf-exc",
#             "taraban_ctrl-hf-reg-inc",
#             "taraban_ctrl-lf-exc",
#             "taraban_ctrl-lf-reg-inc",
#         )

#         df = pd.DataFrame()

#         for testset_name in testsets:
#             df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "taraban_item_df.csv"))

#         mean_df = (
#             df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
#             .mean()
#             .reset_index()
#         )

#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", "taraban_mean_df.csv")
#         )

#         self.taraban_mean_df = mean_df

#         return df


# class EvalReading:
#     """Bundle of testsets"""

#     TESTSETS_NAME = ("strain", "grain")

#     def __init__(self, cfg, model, data):
#         self.cfg = cfg
#         self.model = model
#         self.data = data

#         self.strain_mean_df = None
#         self.grain_mean_df = None

#         # Load eval results from file
#         for _testset_name in self.TESTSETS_NAME:
#             try:
#                 _file = os.path.join(
#                     self.cfg.model_folder,
#                     "eval",
#                     f"{_testset_name}_mean_df.csv",
#                 )
#                 setattr(self, f"{_testset_name}_mean_df", pd.read_csv(_file))
#             except (FileNotFoundError, IOError):
#                 pass

#         # Bundle testsets into dictionary
#         self.run_eval = {
#             "train": self._eval_train,
#             "strain": self._eval_strain,
#             "grain": self._eval_grain,
#             "taraban": self._eval_taraban,
#             "cortese": self._eval_cortese,
#         }

#     def eval(self, testset_name):
#         """Run eval and push to dat"""
#         if getattr(self, f"{testset_name}_mean_df") is None:
#             results = self.run_eval[testset_name]()
#         else:
#             print("Evaluation results found, loaded from file.")

#     def _eval_train(self):
#         testset_name = "train"
#         t = TestSet(
#             name=testset_name,
#             cfg=self.cfg,
#             model=self.model,
#             task="triangle",
#             testitems=self.data.testsets[testset_name]["item"],
#             x_test=self.data.testsets[testset_name]["ort"],
#             y_test=[
#                 self.data.testsets[testset_name]["pho"],
#                 self.data.testsets[testset_name]["sem"],
#             ],
#         )
#         t.eval_all()
#         df = t.result
#         df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", f"{testset_name}_item_df.csv")
#         )

#         # Aggregate
#         mean_df = (
#             df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
#             .mean()
#             .reset_index()
#         )
#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", f"{testset_name}_mean_df.csv")
#         )

#         self.train_mean_df = mean_df

#         return df

#     def _eval_strain(self):

#         df = pd.DataFrame()
#         testsets = (
#             "strain_hf_con_hi",
#             "strain_hf_inc_hi",
#             "strain_hf_con_li",
#             "strain_hf_inc_li",
#             "strain_lf_con_hi",
#             "strain_lf_inc_hi",
#             "strain_lf_con_li",
#             "strain_lf_inc_li",
#         )

#         for testset_name in testsets:
#             t = TestSet(
#                 name=testset_name,
#                 cfg=self.cfg,
#                 model=self.model,
#                 task="triangle",
#                 testitems=self.data.testsets[testset_name]["item"],
#                 x_test=self.data.testsets[testset_name]["ort"],
#                 y_test=[
#                     self.data.testsets[testset_name]["pho"],
#                     self.data.testsets[testset_name]["sem"],
#                 ],
#             )

#             t.eval_all()
#             df = pd.concat([df, t.result])

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "strain_item_df.csv"))

#         # Condition level aggregate
#         mean_df = (
#             df.groupby(
#                 [
#                     "code_name",
#                     "task",
#                     "testset",
#                     "epoch",
#                     "timetick",
#                     "y",
#                 ]
#             )
#             .mean()
#             .reset_index()
#         )
#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", "strain_mean_df.csv")
#         )
#         self.strain_mean_df = mean_df

#         return df

#     def _eval_grain(self):
#         df = pd.DataFrame()
#         for testset_name in ("grain_unambiguous", "grain_ambiguous"):
#             for grain_size in ("pho_small_grain", "pho_large_grain"):
#                 t = TestSet(
#                     name=testset_name,
#                     cfg=self.cfg,
#                     model=self.model,
#                     task="triangle",
#                     testitems=self.data.testsets[testset_name]["item"],
#                     x_test=self.data.testsets[testset_name]["ort"],
#                     y_test=[
#                         self.data.testsets[testset_name][grain_size],
#                         self.data.testsets[testset_name]["sem"],
#                     ],
#                 )

#                 t.eval_all()
#                 t.result["y_test"] = grain_size
#                 df = pd.concat([df, t.result])

#         # Pho only
#         pho_df = df.loc[df.y == "pho"]

#         # Calculate pho acc by summing large and small grain response
#         pho_acc_df = (
#             pho_df.groupby(
#                 ["code_name", "task", "y", "testset", "epoch", "timetick", "item"]
#             )
#             .sum()
#             .reset_index()
#         )

#         pho_acc_df["y_test"] = "pho"

#         # Sem only (Because we have evaluated semantic twice, we need to remove the duplicates)
#         sem_df = df.loc[(df.y == "sem") & (df.y_test == "pho_small_grain")]
#         sem_df = sem_df.drop(columns="y_test")
#         sem_df["y_test"] = "sem"

#         df = pd.concat([pho_df, pho_acc_df, sem_df])
#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "grain_item_df.csv"))

#         mean_df = (
#             df.groupby(
#                 ["code_name", "task", "testset", "epoch", "timetick", "y", "y_test"]
#             )
#             .mean()
#             .reset_index()
#         )
#         mean_df.to_csv(os.path.join(self.cfg.model_folder, "eval", "grain_mean_df.csv"))

#         self.grain_mean_df = mean_df

#         return df

#     def _eval_taraban(self):

#         testsets = (
#             "taraban_hf-exc",
#             "taraban_hf-reg-inc",
#             "taraban_lf-exc",
#             "taraban_lf-reg-inc",
#             "taraban_ctrl-hf-exc",
#             "taraban_ctrl-hf-reg-inc",
#             "taraban_ctrl-lf-exc",
#             "taraban_ctrl-lf-reg-inc",
#         )

#         df = pd.DataFrame()

#         for testset_name in testsets:

#             t = TestSet(
#                 name=testset_name,
#                 cfg=self.cfg,
#                 model=self.model,
#                 task="triangle",
#                 testitems=self.data.testsets[testset_name]["item"],
#                 x_test=self.data.testsets[testset_name]["ort"],
#                 y_test=[
#                     self.data.testsets[testset_name]["pho"],
#                     self.data.testsets[testset_name]["sem"],
#                 ],
#             )

#             t.eval_all()
#             df = pd.concat([df, t.result])

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "taraban_item_df.csv"))

#         mean_df = (
#             df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
#             .mean()
#             .reset_index()
#         )

#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", "taraban_mean_df.csv")
#         )

#         self.taraban_mean_df = mean_df

#         return df

#     def _eval_cortese(self):

#         df = pd.DataFrame()
#         for testset_name in ("cortese_hi_img", "cortese_low_img"):
#             t = TestSet(
#                 name=testset_name,
#                 cfg=self.cfg,
#                 model=self.model,
#                 task="triangle",
#                 testitems=self.data.testsets[testset_name]["item"],
#                 x_test=self.data.testsets[testset_name]["ort"],
#                 y_test=[
#                     self.data.testsets[testset_name]["pho"],
#                     self.data.testsets[testset_name]["sem"],
#                 ],
#             )

#             t.eval_all()
#             df = pd.concat([df, t.result])

#         df.to_csv(os.path.join(self.cfg.model_folder, "eval", "cortese_item_df.csv"))

#         mean_df = (
#             df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
#             .mean()
#             .reset_index()
#         )

#         mean_df.to_csv(
#             os.path.join(self.cfg.model_folder, "eval", "cortese_mean_df.csv")
#         )

#         self.cortese_mean_df = mean_df

#         return df

#     def plot_reading_acc(self, df):
#         timetick_selection = alt.selection_single(
#             bind=alt.binding_range(min=0, max=self.cfg.n_timesteps, step=1),
#             fields=["timetick"],
#             init={"timetick": self.cfg.n_timesteps},
#             name="timetick",
#         )

#         p = (
#             alt.Chart(df)
#             .mark_line()
#             .encode(
#                 x="epoch:Q",
#                 y=alt.Y("acc:Q", scale=alt.Scale(domain=(0, 1))),
#                 color="y",
#             )
#             .add_selection(timetick_selection)
#             .transform_filter(timetick_selection)
#         )

#         return p

#     def plot_grain_by_resp(self):
#         df = self.grain_mean_df.loc[
#             self.grain_mean_df.y_test.isin(["pho_large_grain", "pho_small_grain"])
#         ]
#         p = self.plot_reading_acc(df).encode(color="testset", strokeDash="y_test")
#         return p


# # New set of plots


# def plot_reading_acc(df):
#     timetick_selection = alt.selection_single(
#         bind=alt.binding_range(min=0, max=12, step=1),
#         fields=["timetick"],
#         init={"timetick": 12},
#         name="timetick",
#     )

#     y_selection = alt.selection_single(
#         bind=alt.binding_select(options=["pho", "sem"]),
#         fields=["y"],
#     )

#     return (
#         alt.Chart(df)
#         .mark_line()
#         .encode(
#             x="epoch:Q",
#             y=alt.Y("mean(acc):Q", scale=alt.Scale(domain=(0, 1))),
#             color="testset",
#         )
#         .add_selection(timetick_selection)
#         .add_selection(y_selection)
#         .transform_filter(timetick_selection)
#         .transform_filter(y_selection)
#     )


# def plot_grain_acceptable(df):

#     df = df.loc[df.y_test.isin(["pho", "sem"]) & (df.y == "pho")]

#     timetick_selection = alt.selection_single(
#         bind=alt.binding_range(min=0, max=cfg.n_timesteps, step=1),
#         fields=["timetick"],
#         init={"timetick": cfg.n_timesteps},
#         name="timetick",
#     )

#     return (
#         alt.Chart(df)
#         .mark_line()
#         .encode(
#             x="epoch:Q",
#             y=alt.Y("acc:Q", scale=alt.Scale(domain=(0, 1))),
#             color="testset",
#         )
#         .add_selection(timetick_selection)
#         .transform_filter(timetick_selection)
#     )


# def plot_grain_by_resp(df):
#     df = df.loc[df.y_test.isin(["pho_large_grain", "pho_small_grain"])]
#     p = plot_reading_acc(df).encode(color="testset", strokeDash="y_test")
#     return p


# def plot_contrast(df, use_y="acc", y_max=1, task="triangle"):
#     df = df.loc[(df.task == task)]

#     timetick_selection = alt.selection_single(
#         bind=alt.binding_range(min=0, max=12, step=1),
#         fields=["timetick"],
#         init={"timetick": cfg.n_timesteps},
#         name="timetick",
#     )

#     y_selection = alt.selection_single(
#         bind=alt.binding_select(options=["pho", "sem"]),
#         init={"y": "pho"},
#         fields=["y"],
#     )

#     cond_selection = alt.selection_multi(bind="legend", fields=["testset"])

#     # Plot by condition
#     plot_by_cond = (
#         alt.Chart(df)
#         .mark_line()
#         .encode(
#             x="epoch:Q",
#             y=alt.Y(f"{use_y}:Q", scale=alt.Scale(domain=(0, y_max))),
#             color="testset:N",
#             opacity=alt.condition(cond_selection, alt.value(1), alt.value(0.1)),
#         )
#         .add_selection(timetick_selection)
#         .add_selection(y_selection)
#         .add_selection(cond_selection)
#         .transform_filter(timetick_selection)
#         .transform_filter(y_selection)
#     )

#     # Plot contrasts
#     contrasts = {}
#     contrasts[
#         "F contrast"
#     ] = """(datum.strain_hf_con_hi + datum.strain_hf_con_li + datum.strain_hf_inc_hi + datum.strain_hf_inc_li -
#         (datum.strain_lf_con_hi + datum.strain_lf_con_li + datum.strain_lf_inc_hi + datum.strain_lf_inc_li))/4"""
#     contrasts[
#         "CON contrast"
#     ] = """(datum.strain_hf_con_hi + datum.strain_hf_con_li + datum.strain_lf_con_hi + datum.strain_lf_con_li -
#         (datum.strain_hf_inc_hi + datum.strain_hf_inc_li + datum.strain_lf_inc_hi + datum.strain_lf_inc_li))/4"""
#     contrasts[
#         "IMG contrast"
#     ] = """(datum.strain_hf_con_hi + datum.strain_lf_con_hi + datum.strain_hf_inc_hi + datum.strain_lf_inc_hi -
#         (datum.strain_hf_con_li + datum.strain_lf_con_li + datum.strain_hf_inc_li + datum.strain_lf_inc_li))/4"""

#     def create_contrast_plot(name):
#         return (
#             plot_by_cond.encode(
#                 y=alt.Y("difference:Q", scale=alt.Scale(domain=(-y_max, y_max)))
#             )
#             .transform_pivot("testset", value=use_y, groupby=["epoch"])
#             .transform_calculate(difference=contrasts[name])
#             .properties(title=name, width=100, height=100)
#         )

#     contrast_plots = alt.hconcat()
#     for c in contrasts.keys():
#         contrast_plots |= create_contrast_plot(c)

#     return plot_by_cond | contrast_plots
