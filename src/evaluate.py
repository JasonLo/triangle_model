from altair.vegalite.v4.schema.channels import StrokeDash
from tqdm import tqdm
import os
import metrics
import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()


class testset:
    """Universal test set object for evaluating model results
    1. Single condition, single metric, single value output for maximum capatibility
    2. Model level info should be stored at separate table, and merge it at the end
    """

    def __init__(
        self,
        name,
        cfg,
        model,
        task,
        testitems,
        x_test,
        y_test,
        metric,
        triangle_out=None,
    ):
        self.name = name
        self.cfg = cfg
        self.model = model
        self.task = task
        self.model.set_active_task(self.task)
        self.testitems = testitems
        self.x_test = x_test
        self.y_test = y_test
        self.metric = metric
        self.triangle_out = triangle_out

    def _convert_dict_to_df(self, x):

        self.flat_dict = {
            (epoch, y, timetick, item): {metric: x[epoch][y][timetick][metric][item]}
            for epoch in x.keys()
            for y in x[epoch].keys()
            for timetick in x[epoch][y].keys()
            for metric in x[epoch][y][timetick].keys()
            for item in x[epoch][y][timetick][metric].keys()
        }

        df = pd.DataFrame.from_dict(self.flat_dict, orient="index")
        df.index.rename(["epoch", "y", "timetick", "item"], inplace=True)
        df.reset_index(inplace=True)
        return df

    def eval_all(self, label_dict=None):
        output = {}
        for epoch in tqdm(self.cfg.saved_epoches, desc=f"Evaluating {self.name}"):
            output[epoch] = self._eval_one_epoch(epoch)

        self.outputdict = output
        df = self._convert_dict_to_df(output)
        df["code_name"] = self.cfg.code_name
        df["testset"] = self.name
        df["task"] = self.task

        try:
            for k, v in label_dict.items():
                df[k] = v
        except:
            pass

        self.result = df

    def _eval_one_epoch(self, epoch):
        checkpoint = self.cfg.path["weights_checkpoint_fstring"].format(epoch=epoch)
        self.model.load_weights(checkpoint)

        pred_y = self.model([self.x_test] * self.cfg.n_timesteps)

        output = {}
        if self.task == "triangle":
            output["pho"] = self._eval_one_y(pred_y[0], self.y_test[0])
            output["sem"] = self._eval_one_y(pred_y[1], self.y_test[1])
        elif (self.task == "pho_sem") or (self.task == "sem_sem"):              
            output["sem"] = self._eval_one_y(pred_y)
        elif (self.task == "sem_pho") or (self.task == "pho_pho"):
            output["pho"] = self._eval_one_y(pred_y)
        else:
            print(f"{self.task} task does not exist in evaluator")
 
        return output
    
    def _eval_one_y(self, pred_y, true_y):
        output = {}
        if type(pred_y) is list:
            # Model with multi time ticks
            for i, pred_y_at_this_time in enumerate(pred_y):
                tick = self.cfg.n_timesteps - self.cfg.output_ticks + i + 1
                output[tick] = self._eval_one_timetick(pred_y_at_this_time, true_y)
        else:
            # Model with only one output tick
            output[self.cfg.n_timesteps] = self._eval_one_timetick(pred_y, true_y)
        return output
    
    def _eval_one_timetick(self, pred_y, true_y):

        output = {}
        output[self.metric.name] = dict(
            zip(self.testitems, self.metric.item_metric(true_y, pred_y))
        )

        return output


class eval_reading:
    """Bundle of testsets"""

    Y_CONFIG_DICT = {
        "pho": {"triangle_out": "pho", "metric": metrics.PhoAccuracy("acc")},
        "pho_large_grain": {
            "triangle_out": "pho",
            "metric": metrics.PhoAccuracy("acc"),
        },
        "pho_small_grain": {
            "triangle_out": "pho",
            "metric": metrics.PhoAccuracy("acc"),
        },
        "sem": {"triangle_out": "sem", "metric": metrics.RightSideAccuracy("acc")},
    }

    def __init__(self, cfg, model, data):
        self.cfg = cfg
        self.model = model
        self.data = data

    def run_eval(self, testset_name, ys=["pho", "sem"]):

        output = pd.DataFrame()
        for y in ys:

            acc = testset(
                name=f"{testset_name}",
                cfg=self.cfg,
                model=self.model,
                task="triangle",
                triangle_out=self.Y_CONFIG_DICT[y]["triangle_out"],
                testitems=self.data.testsets[testset_name]["item"],
                x_test=self.data.testsets[testset_name]["ort"],
                y_test=self.data.testsets[testset_name][y],
                metric=self.Y_CONFIG_DICT[y]["metric"],
            )
            

            acc.eval_all()
            acc.result["y_test"] = y

            output = pd.concat([output, tmp.result])

        return output

    def eval_train(self):
        """Call run_eval in testset (train), and run testset specific post-processing"""

        df = self.run_eval("train")
        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_item_train.csv"))
        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y_test"])
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_train.csv")
        )
        self.train_mean_df = mean_df

    def eval_strain(self):
        """Call run_eval in testset (strain), and run testset specific post-processing"""

        df = self.run_eval("strain")
        df = df.merge(
            self.data.df_strain[
                ["word", "frequency", "pho_consistency", "imageability"]
            ],
            how="left",
            left_on="item",
            right_on="word",
        )

        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_item_strain.csv"))
        mean_df = (
            df.groupby(
                [
                    "code_name",
                    "task",
                    "testset",
                    "epoch",
                    "timetick",
                    "y_test",
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
        for g in ("grain_unambiguous", "grain_ambiguous"):
            tmp = self.run_eval(g, ys=["pho_small_grain", "pho_large_grain", "sem"])
            df = pd.concat([df, tmp])

        # Calculate pho acc by summing large and small grain response
        acc_df = (
            df.loc[df.y_test.isin(["pho_large_grain", "pho_small_grain"])]
            .groupby(["code_name", "task", "testset", "epoch", "timetick", "item"])
            .sum()
            .reset_index()
        )
        acc_df["y_test"] = "pho"
        df = pd.concat([df, acc_df])
        df.to_csv(os.path.join(self.cfg.path["model_folder"], "eval_item_grain.csv"))

        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y_test"])
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval_mean_grain.csv")
        )

        self.grain_mean_df = mean_df

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
                color="y_test",
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
