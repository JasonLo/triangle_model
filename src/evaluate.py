""" This module is for all things related to testsets
"""

from tqdm import tqdm
import os
import metrics
import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()

class TestSet:
    """Universal test set object for evaluating model results
    1. Single condition, single metric, single value output for maximum capatibility
    2. Model level info should be stored at separate table, and merge it at the end
    """

    pho_acc = metrics.PhoAccuracy("acc")
    right_side_acc = metrics.RightSideAccuracy("acc")
    sse = metrics.SumSquaredError("sse")

    def __init__(
        self,
        name,
        cfg,
        model,
        task,
        testitems,
        x_test,
        y_test,
    ):
        self.name = name
        self.cfg = cfg
        self.model = model
        self.task = task
        self.model.set_active_task(self.task)
        self.testitems = testitems
        self.x_test = x_test
        self.y_test = y_test

        self._flat_dict = None
        self.result = None

    def _convert_dict_to_df(self, x):

        # Flatten the nested output dictionary
        self._flat_dict = {
            (epoch, y, timetick, item, metric): {
                "value": x[epoch][y][timetick][metric][item]
            }
            for epoch in x.keys()
            for y in x[epoch].keys()
            for timetick in x[epoch][y].keys()
            for metric in x[epoch][y][timetick].keys()
            for item in x[epoch][y][timetick][metric].keys()
        }

        # Create pd df and pivot by metric as column
        df = pd.DataFrame.from_dict(self._flat_dict, orient="index")
        df.index.rename(["epoch", "y", "timetick", "item", "metric"], inplace=True)
        df = df.pivot_table(
            index=["epoch", "timetick", "y", "item"],
            columns="metric",
            values="value",
        ).reset_index()

        return df

    def eval_all(self, label_dict=None):
        output = {}
        for epoch in tqdm(self.cfg.saved_epoches, desc=f"Evaluating {self.name}"):
            output[epoch] = self._eval_one_epoch(epoch)

        df = self._convert_dict_to_df(output)
        df["code_name"] = self.cfg.code_name
        df["testset"] = self.name
        df["task"] = self.task

        # Attach additional labels
        try:
            for k, v in label_dict.items():
                df[k] = v
        except AttributeError:
            pass

        self.result = df

    def _eval_one_epoch(self, epoch):
        checkpoint = self.cfg.path["weights_checkpoint_fstring"].format(epoch=epoch)
        self.model.load_weights(checkpoint)

        pred_y = self.model([self.x_test] * self.cfg.n_timesteps)

        output = {}
        if self.task == "triangle":
            output["pho"] = self._eval_one_y(pred_y[0], self.y_test[0], y_name="pho")
            output["sem"] = self._eval_one_y(pred_y[1], self.y_test[1], y_name="sem")
        elif (self.task == "pho_sem") or (self.task == "sem_sem"):
            output["sem"] = self._eval_one_y(pred_y, self.y_test, y_name="sem")
        elif (self.task == "sem_pho") or (self.task == "pho_pho"):
            output["pho"] = self._eval_one_y(pred_y, self.y_test, y_name="pho")
        else:
            print(f"{self.task} task does not exist in evaluator")

        return output

    def _eval_one_y(self, pred_y, true_y, y_name):
        output = {}
        if type(pred_y) is list:
            # Model with multi time ticks
            for i, pred_y_at_this_time in enumerate(pred_y):
                tick = self.cfg.n_timesteps - self.cfg.output_ticks + i + 1
                output[tick] = self._eval_one_timetick(
                    pred_y_at_this_time, true_y, y_name
                )
        else:
            # Model with only one output tick
            output[self.cfg.n_timesteps] = self._eval_one_timetick(
                pred_y, true_y, y_name
            )
        return output

    def _eval_one_timetick(self, pred_y, true_y, y_name):

        output = {}

        # Use different acc depending on output nature
        if y_name == "pho":
            output["acc"] = dict(
                zip(self.testitems, self.pho_acc.item_metric(true_y, pred_y))
            )
        elif y_name == "sem":
            output["acc"] = dict(
                zip(self.testitems, self.right_side_acc.item_metric(true_y, pred_y))
            )

        output["sse"] = dict(zip(self.testitems, self.sse.item_metric(true_y, pred_y)))
        return output
    
    

class EvalReading:
    """Bundle of testsets"""

    def __init__(self, cfg, model, data):
        self.cfg = cfg
        self.model = model
        self.data = data

    def eval_train(self):
        testset_name = "train"
        t = TestSet(
            name=testset_name,
            cfg=self.cfg,
            model=self.model,
            task="triangle",
            testitems=self.data.testsets[testset_name]["item"],
            x_test=self.data.testsets[testset_name]["ort"],
            y_test=[
                self.data.testsets[testset_name]["pho"],
                self.data.testsets[testset_name]["sem"],
            ],
        )

        t.eval_all()
        df = t.result

        # Item level
        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_item_train.csv"))

        # Aggregate
        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
            .mean()
            .reset_index()
        )

        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_train.csv")
        )
        self.train_mean_df = mean_df

    def eval_strain(self):
        testset_name = "strain"

        t = TestSet(
            name=testset_name,
            cfg=self.cfg,
            model=self.model,
            task="triangle",
            testitems=self.data.testsets[testset_name]["item"],
            x_test=self.data.testsets[testset_name]["ort"],
            y_test=[
                self.data.testsets[testset_name]["pho"],
                self.data.testsets[testset_name]["sem"],
            ],
        )

        t.eval_all()
        df = t.result

        # Merge condition label
        df = df.merge(
            self.data.df_strain[
                ["word", "frequency", "pho_consistency", "imageability"]
            ],
            how="left",
            left_on="item",
            right_on="word",
        )

        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_item_strain.csv"))

        # Condition level aggregate
        mean_df = (
            df.groupby(
                [
                    "code_name",
                    "task",
                    "testset",
                    "epoch",
                    "timetick",
                    "y",
                    "frequency",
                    "pho_consistency",
                    "imageability",
                ]
            )
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_strain.csv")
        )
        self.strain_mean_df = mean_df

    def eval_grain(self):
        """Call run_eval in testset (grain), and run testset specific post-processing"""

        df = pd.DataFrame()
        for testset_name in ("grain_unambiguous", "grain_ambiguous"):
            for grain_size in ("pho_small_grain", "pho_large_grain"):
                t = TestSet(
                    name=testset_name,
                    cfg=self.cfg,
                    model=self.model,
                    task="triangle",
                    testitems=self.data.testsets[testset_name]["item"],
                    x_test=self.data.testsets[testset_name]["ort"],
                    y_test=[
                        self.data.testsets[testset_name][grain_size],
                        self.data.testsets[testset_name]["sem"],
                    ],
                )

                t.eval_all()
                t.result["y_test"] = grain_size
                df = pd.concat([df, t.result])

        # Pho only
        pho_df = df.loc[df.y == "pho"]

        # Calculate pho acc by summing large and small grain response
        pho_acc_df = (
            pho_df.groupby(
                ["code_name", "task", "y", "testset", "epoch", "timetick", "item"]
            )
            .sum()
            .reset_index()
        )

        pho_acc_df["y_test"] = "pho"

        # Sem only (Because we have evaluated semantic twice, we need to remove the duplicates)
        sem_df = df.loc[(df.y == "sem") & (df.y_test == "pho_small_grain")]
        sem_df = sem_df.drop(columns="y_test")
        sem_df["y_test"] = "sem"

        df = pd.concat([pho_df, pho_acc_df, sem_df])
        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_item_grain.csv"))

        mean_df = (
            df.groupby(
                ["code_name", "task", "testset", "epoch", "timetick", "y", "y_test"]
            )
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_grain.csv")
        )

        self.grain_mean_df = mean_df

    def eval_taraban(self):

        testsets = (
            "taraban_hf-exc",
            "taraban_hf-reg-inc",
            "taraban_lf-exc",
            "taraban_lf-reg-inc",
            "taraban_ctrl-hf-exc",
            "taraban_ctrl-hf-reg-inc",
            "taraban_ctrl-lf-exc",
            "taraban_ctrl-lf-reg-inc"
            )

        df = pd.DataFrame()

        for testset_name in testsets:

            t = TestSet(
                name=testset_name,
                cfg=self.cfg,
                model=self.model,
                task="triangle",
                testitems=self.data.testsets[testset_name]["item"],
                x_test=self.data.testsets[testset_name]["ort"],
                y_test=[
                    self.data.testsets[testset_name]["pho"],
                    self.data.testsets[testset_name]["sem"],
                ],
            )

            t.eval_all()
            df = pd.concat([df, t.result])
        
        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_taraban.csv"))

        mean_df = (
            df.groupby(
                ["code_name", "task", "testset", "epoch", "timetick", "y"]
            )
            .mean()
            .reset_index()
        )

        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_taraban.csv")
        )

        self.taraban_mean_df = mean_df

    def eval_train_cortese_img(self):

        df = pd.DataFrame()
        for testset_name in ("train_cortese_hi_img", "train_cortese_low_img"):
            t = TestSet(
                name=testset_name,
                cfg=self.cfg,
                model=self.model,
                task="triangle",
                testitems=self.data.testsets[testset_name]["item"],
                x_test=self.data.testsets[testset_name]["ort"],
                y_test=[
                    self.data.testsets[testset_name]["pho"],
                    self.data.testsets[testset_name]["sem"],
                ],
            )

            t.eval_all()
            df = pd.concat([df, t.result])

        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_train_cortese_img.csv"))

        mean_df = (
            df.groupby(
                ["code_name", "task", "testset", "epoch", "timetick", "y"]
            )
            .mean()
            .reset_index()
        )

        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_train_cortese_img.csv")
        )

        self.train_cortese_img_mean_df = mean_df

    def plot_reading_acc(self, df):
        timetick_selection = alt.selection_single(
            bind=alt.binding_range(min=0, max=self.cfg.n_timesteps, step=1),
            fields=["timetick"],
            init={"timetick": self.cfg.n_timesteps},
            name="timetick",
        )

        p = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="epoch:Q",
                y=alt.Y("acc:Q", scale=alt.Scale(domain=(0, 1))),
                color="y",
            )
            .add_selection(timetick_selection)
            .transform_filter(timetick_selection)
        )

        return p

    def plot_grain_by_resp(self):
        df = self.grain_mean_df.loc[
            self.grain_mean_df.y_test.isin(["pho_large_grain", "pho_small_grain"])
        ]
        p = self.plot_reading_acc(df).encode(color="testset", strokeDash="y_test")
        return p