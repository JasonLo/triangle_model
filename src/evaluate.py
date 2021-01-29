from tqdm import tqdm
import metrics
import pandas as pd


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
            (epoch, timetick, item): {metric: x[epoch][timetick][metric][item]}
            for epoch in x.keys()
            for timetick in x[epoch].keys()
            for metric in x[epoch][timetick].keys()
            for item in x[epoch][timetick][metric].keys()
        }

        df = pd.DataFrame.from_dict(self.flat_dict, orient="index")
        df.index.rename(["epoch", "timeticks", "item"], inplace=True)
        df.reset_index(inplace=True)
        return df

    def eval_all(self, label_dict=None):
        output = {}
        for epoch in tqdm(self.cfg.saved_epoches, desc=f"Evaluating {self.name}"):
            output[epoch] = self._eval_one_epoch(epoch)

        df = self._convert_dict_to_df(output)
        df["code_name"] = self.cfg.code_name
        df["testset"] = self.name
        df["task"] = self.task

        try:
            df["triangle_out"] = self.triangle_out
            for k, v in label_dict.items():
                df[k] = v
        except:
            pass

        self.result = df

    def _eval_one_epoch(self, epoch):
        checkpoint = self.cfg.path["weights_checkpoint_fstring"].format(epoch=epoch)
        self.model.load_weights(checkpoint)

        pred_y = self.model([self.x_test] * self.cfg.n_timesteps)

        if self.triangle_out is not None:
            if self.triangle_out == "pho":
                pred_y = pred_y[0]
            elif self.triangle_out == "sem":
                pred_y = pred_y[1]

        output = {}
        if type(pred_y) is list:
            for i, pred_y_at_this_time in enumerate(pred_y):
                tick = self.cfg.n_timesteps - self.cfg.output_ticks + i + 1
                output[tick] = self._eval_one_timetick(pred_y_at_this_time)
        else:
            output[self.cfg.n_timesteps] = self._eval_one_timetick(pred_y)

        return output

    def _eval_one_timetick(self, y_pred):

        output = {}
        output[self.metric.name] = dict(
            zip(self.testitems, self.metric.item_metric(self.y_test, y_pred))
        )

        return output


class eval_reading:
    """Bundle of testsets"""
    Y_CONFIG_DICT = {
        "pho": {"triangle_out": "pho", "metric": metrics.PhoAccuracy("acc")},
        "pho_large_grain": {"triangle_out": "pho", "metric": metrics.PhoAccuracy("acc")},
        "pho_small_grain": {"triangle_out": "pho", "metric": metrics.PhoAccuracy("acc")},
        "sem": {"triangle_out": "sem", "metric": metrics.RightSideAccuracy("acc")},
    }

    def __init__(self, cfg, model, data):
        self.cfg = cfg
        self.model = model
        self.data = data

    def run_eval(self, testset_name, ys=["pho", "sem"]):

        output = pd.DataFrame()
        for y in ys:
            
            tmp = testset(
                name=f'{testset_name}_{y}',
                cfg=self.cfg,
                model=self.model,
                task="triangle",
                triangle_out=self.Y_CONFIG_DICT[y]["triangle_out"],
                testitems=self.data.testsets[testset_name]["item"],
                x_test=self.data.testsets[testset_name]["ort"],
                y_test=self.data.testsets[testset_name][y],
                metric=self.Y_CONFIG_DICT[y]["metric"]
            )

            tmp.eval_all()
            output = pd.concat([output, tmp.result])
        
        return output

    def eval_strain(self):

        df = self.run_eval("strain")
        df = df.merge(
            self.data.df_strain[["word", "frequency", "pho_consistency", "imageability"]], 
            how="left", left_on="item", right_on="word")

        df['cond'] = df.frequency + '_' + df.pho_consistency + '_' + df.imageability

        self.strain_df = df
