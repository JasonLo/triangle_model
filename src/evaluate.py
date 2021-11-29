""" This module is for all things related to testsets
"""

from tqdm import tqdm
from meta import Config
import os
import tensorflow as tf
import metrics, modeling, gcp
from data_wrangling import load_testset
from helper import get_batch_pronunciations_fast
import pandas as pd


class Test:
    """Universal test set object for evaluating model results
    1. Single condition, single metric, single value output for maximum compatibility
    2. Model level info should be stored at separate table, and merge it in the end
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = modeling.TriangleModel(self.cfg)
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.metrics = {
            "acc": {"pho": metrics.PhoAccuracy(), "sem": metrics.RightSideAccuracy()},
            "sse": metrics.SumSquaredError(),
            "act0": metrics.OutputOfZeroTarget(),
            "act1": metrics.OutputOfOneTarget(),
        }

    def eval_train(self, task: str, n: int = 12, bq_table: str = None):
        """Evaluate the full training set with manual batching.
        Due to memory demands, will save results to csv file
        """
        [self.eval(f"train_batch_{i}", task, bq_table=bq_table) for i in range(n)]

    def eval(
        self,
        testset_name: str,
        task: str,
        save_file_prefix: str = None,
        bq_table: str = None,
    ):
        """
        Inputs
        testset_name: name of testset, must match testset package (*.pkl.gz) name
        task: task name option in triangle model
        output: pandas dataframe with all the evaluation results
        """
        try:
            # Try to load from saved eval csv
            df = self.load(testset_name, task, save_file_prefix)
            print(f"Eval results found, load from saved csv")

        except (FileNotFoundError, IOError):

            df = pd.DataFrame()
            testset_package = load_testset(testset_name)

            # Enforceing batch_size dim to match with test case
            inputs = testset_package[modeling.IN_OUT[task][0]]

            # Build model and switch task
            self.model.set_active_task(task)

            for epoch in tqdm(
                self.cfg.saved_epochs, desc=f"Evaluating {testset_name}:{task}"
            ):
                saved_checkpoint = self.cfg.saved_checkpoints_fstring.format(
                    epoch=epoch
                )
                self.ckpt.restore(saved_checkpoint).expect_partial()
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

            if bq_table:
                gcp.df_to_bigquery(df, self.cfg.batch_name, bq_table)
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
