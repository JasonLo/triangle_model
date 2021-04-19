""" This module is for all things related to testsets
"""

from tqdm import tqdm
import os
import metrics
import sqlite3
import pandas as pd
import numpy as np
import altair as alt

alt.data_transformers.disable_max_rows()

def gen_pkey(p_file="/home/jupyter/tf/dataset/mappingv2.txt"):
    """Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict("list")
    return m_dict

phon_key = gen_pkey()

def get_pronunciation_fast(act, phon_key):
    phonemes = list(phon_key.keys())
    act10 = np.tile([v for k, v in phon_key.items()], 10)

    d = np.abs(act10 - act)
    d_mat = np.reshape(d, (38, 10, 25))
    sumd_mat = np.squeeze(np.sum(d_mat, 2))
    map_idx = np.argmin(sumd_mat, 0)
    out = str()
    for x in map_idx:
        out += phonemes[x]
    return out

def get_batch_pronunciations_fast(act, phon_key):
    return np.apply_along_axis(get_pronunciation_fast, 1, act, phon_key)



class TestSet:
    """Universal test set object for evaluating model results
    1. Single condition, single metric, single value output for maximum capatibility
    2. Model level info should be stored at separate table, and merge it at the end
    """

    pho_acc = metrics.PhoAccuracy("acc")
    right_side_acc = metrics.RightSideAccuracy("acc")
    sse = metrics.SumSquaredError("sse")
    act1 = metrics.OutputOfOneTarget("act1")
    act0 = metrics.OutputOfZeroTarget("act0")

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

        df = df.reset_index().pivot(
            index=["epoch", "timetick", "y", "item"],
            columns="metric",
            values="value",
        ).reset_index()
        
        # automatically convert to best dtype
        return df.convert_dtypes()

    def eval_all(self, label_dict=None):       
        df = pd.DataFrame()
        for epoch in tqdm(self.cfg.saved_epoches, desc=f"Evaluating {self.name}"):
            output = {}
            output[epoch] = self._eval_one_epoch(epoch)
            this_epoch_df = self._convert_dict_to_df(output)
            df = df.append(this_epoch_df, ignore_index=True)

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
        if (self.task in ("triangle")):
            output["pho"] = self._eval_one_y(pred_y[0], self.y_test[0], y_name="pho")
            output["sem"] = self._eval_one_y(pred_y[1], self.y_test[1], y_name="sem")
        elif (self.task in ("pho_sem", "sem_sem", "ort_sem")):
            output["sem"] = self._eval_one_y(pred_y, self.y_test, y_name="sem")
        elif (self.task in ("sem_pho", "pho_pho", "ort_pho")):
            output["pho"] = self._eval_one_y(pred_y, self.y_test, y_name="pho")
        elif (self.task in ("exp_osp")):
            output["pho"] = self._eval_one_y(pred_y["act_p"], self.y_test[0], y_name="pho")
            output["sem"] = self._eval_one_y(pred_y["act_s"], self.y_test[1], y_name="sem")
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
            acc = self.pho_acc.item_metric(true_y, pred_y)
            output["acc"] = dict(
                zip(self.testitems, acc)
            )
            output["pho_symbol"] = dict(
                zip(self.testitems, get_batch_pronunciations_fast(pred_y, phon_key))
            )
        elif y_name == "sem":
            acc = self.right_side_acc.item_metric(true_y, pred_y)
            output["acc"] = dict(
                zip(self.testitems, acc)
            )

        # SSE
        sse = self.sse.item_metric(true_y, pred_y)
        output["sse"] = dict(zip(self.testitems, sse))

        conditional_sse = sse
        conditional_sse[acc==0] = np.nan
        output["conditional_sse"] = dict(zip(self.testitems, conditional_sse))
        
        # Activations
        this_act0 = self.act0.item_metric(true_y, pred_y)
        this_act1 = self.act1.item_metric(true_y, pred_y)
        output["act0"] = dict(zip(self.testitems, this_act0))
        output["act1"] = dict(zip(self.testitems, this_act1))
        
        return output

class EvalOral:
    """Bundle of testsets for Oral stage
    Not finished... only have train strain and taraban cortese img
    """

    TESTSETS_NAME = ("train",  "strain", "taraban")

    def __init__(self, cfg, model, data):
        self.cfg = cfg
        self.model = model
        self.data = data

        self.train_mean_df = None
        self.strain_mean_df = None
        self.grain_mean_df = None
        self.taraban_mean_df = None
        self.cortese_mean_df = None
        self.cortese_img_mean_df = None

        # Setup database
        if self.cfg.batch_name is not None:

            sqlite_file = os.path.join(
                self.cfg.path["batch_folder"], "batch_results.sqlite"
            )
            self.con = sqlite3.connect(sqlite_file)
            self.cur = self.con.cursor()

        # Load eval results from file
        for _testset_name in self.TESTSETS_NAME:
            try:
                _file = os.path.join(
                    self.cfg.path["model_folder"],
                    "eval",
                    f"{_testset_name}_mean_df.csv",
                )
                setattr(self, f"{_testset_name}_mean_df", pd.read_csv(_file))
            except (FileNotFoundError, IOError):
                pass

        # Bundle testsets into dictionary
        self.run_eval = {
            "train": self._eval_train,
            "strain": self._eval_strain,
            "taraban": self._eval_taraban,
            "cortese_img": self._eval_img 
        }

    def eval(self, testset_name):
        """Run eval and push to dat"""
        if getattr(self, f"{testset_name}_mean_df") is None:
            results = self.run_eval[testset_name]()
            try:
                results.to_sql(testset_name, self.con, if_exists="append")
            except:
                pass
        else:
            print("Evaluation results found, loaded from file.")
                  


    def _eval_oral_tasks(self, testset_name):
        """ The oral evalution consists of multiple tasks, sp, ps, pp, ss
        This function will:
        1. Create the four tasks (TestSet object) based on testset_name
        2. Evaluate the tasks
        3. Concatenate all results into output df
        """

        df = pd.DataFrame()

        tasks = ("pho_sem", "sem_pho", "sem_sem", "pho_pho")

        for this_task in tasks:

            x, y = this_task.split('_')
            this_testset_object = TestSet(
                name=testset_name,
                cfg=self.cfg,
                model=self.model,
                task=this_task,
                testitems=self.data.testsets[testset_name]["item"],
                x_test=self.data.testsets[testset_name][x],
                y_test=self.data.testsets[testset_name][y],
            )

            this_testset_object.eval_all()
            df = df.append(this_testset_object.result, ignore_index=True)

        return df

    # Different _eval_xxx bundle has slightly differnt post-processing needs,
    # separate into multiple functions
    def _eval_train(self):
        testset_name = "train"
        df = self._eval_oral_tasks(testset_name)

        # Write item level results
        df.to_csv(
            os.path.join(
                self.cfg.path["model_folder"], "eval", f"{testset_name}_item_df.csv"
            )
        )

        # Aggregate
        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(
                self.cfg.path["model_folder"], "eval", f"{testset_name}_mean_df.csv"
            )
        )

        self.train_mean_df = mean_df

        return df


    
    def _eval_img(self):
        df = pd.DataFrame()
        testsets = ("cortese_hi_img", "cortese_low_img")
        
        for testset_name in testsets:
            df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)
            
        df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "img_item_df.csv")
        )
        
        return df
        
    
    def _eval_strain(self):
        df = pd.DataFrame()
        testsets = (
            "strain_hf_con_hi",
            "strain_hf_inc_hi",
            "strain_hf_con_li",
            "strain_hf_inc_li",
            "strain_lf_con_hi",
            "strain_lf_inc_hi",
            "strain_lf_con_li",
            "strain_lf_inc_li",
        )

        for testset_name in testsets:
            df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)

        df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "strain_item_df.csv")
        )

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
                ]
            )
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "strain_mean_df.csv")
        )
        self.strain_mean_df = mean_df

        return df

    def _eval_taraban(self):

        testsets = (
            "taraban_hf-exc",
            "taraban_hf-reg-inc",
            "taraban_lf-exc",
            "taraban_lf-reg-inc",
            "taraban_ctrl-hf-exc",
            "taraban_ctrl-hf-reg-inc",
            "taraban_ctrl-lf-exc",
            "taraban_ctrl-lf-reg-inc",
        )

        df = pd.DataFrame()

        for testset_name in testsets:
            df = df.append(self._eval_oral_tasks(testset_name), ignore_index=True)

        df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "taraban_item_df.csv")
        )

        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
            .mean()
            .reset_index()
        )

        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "taraban_mean_df.csv")
        )

        self.taraban_mean_df = mean_df

        return df

class EvalReading:
    """Bundle of testsets"""

    TESTSETS_NAME = ("train", "strain", "grain", "taraban", "cortese")

    def __init__(self, cfg, model, data):
        self.cfg = cfg
        self.model = model
        self.data = data

        self.train_mean_df = None
        self.strain_mean_df = None
        self.grain_mean_df = None
        self.taraban_mean_df = None
        self.cortese_mean_df = None
        
        # Setup database if in batch_mode
        if self.cfg.batch_name is not None:
            self.batch_mode = True
            sqlite_file = os.path.join(self.cfg.path["batch_folder"], "batch_results.sqlite")
            self.con = sqlite3.connect(sqlite_file)
        else:
            self.batch_mode = False
        
        # Load eval results from file
        for _testset_name in self.TESTSETS_NAME:
            try:
                _file = os.path.join(
                    self.cfg.path["model_folder"],
                    "eval",
                    f"{_testset_name}_mean_df.csv",
                )
                setattr(self, f"{_testset_name}_mean_df", pd.read_csv(_file))
            except (FileNotFoundError, IOError):
                pass

        # Bundle testsets into dictionary
        self.run_eval = {
            "train": self._eval_train,
            "strain": self._eval_strain,
            "grain": self._eval_grain,
            "taraban": self._eval_taraban,
            "cortese": self._eval_cortese,
        }
        
    def eval(self, testset_name):
        """Run eval and push to dat"""
        if getattr(self, f"{testset_name}_mean_df") is None:
            results = self.run_eval[testset_name]()
     
            if self.batch_mode:
                results.to_sql(testset_name, self.con, if_exists="append")
        else:
            print("Evaluation results found, loaded from file.")
                  
        

    def _eval_train(self):
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
        df.to_csv(
            os.path.join(
                self.cfg.path["model_folder"], "eval", f"{testset_name}_item_df.csv"
            )
        )

        # Aggregate
        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(
                self.cfg.path["model_folder"], "eval", f"{testset_name}_mean_df.csv"
            )
        )

        self.train_mean_df = mean_df
        
        return df

    def _eval_strain(self):

        df = pd.DataFrame()
        testsets = (
            "strain_hf_con_hi",
            "strain_hf_inc_hi",
            "strain_hf_con_li",
            "strain_hf_inc_li",
            "strain_lf_con_hi",
            "strain_lf_inc_hi",
            "strain_lf_con_li",
            "strain_lf_inc_li"
        )

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


        df.to_csv(
            os.path.join(
                self.cfg.path["model_folder"], "eval", "strain_item_df.csv"
            )
        )

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
                ]
            )
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(
                self.cfg.path["model_folder"], "eval", "strain_mean_df.csv"
            )
        )
        self.strain_mean_df = mean_df
        
        return df

    def _eval_grain(self):
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
        df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "grain_item_df.csv")
        )

        mean_df = (
            df.groupby(
                ["code_name", "task", "testset", "epoch", "timetick", "y", "y_test"]
            )
            .mean()
            .reset_index()
        )
        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "grain_mean_df.csv")
        )

        self.grain_mean_df = mean_df
        
        return df

    def _eval_taraban(self):

        testsets = (
            "taraban_hf-exc",
            "taraban_hf-reg-inc",
            "taraban_lf-exc",
            "taraban_lf-reg-inc",
            "taraban_ctrl-hf-exc",
            "taraban_ctrl-hf-reg-inc",
            "taraban_ctrl-lf-exc",
            "taraban_ctrl-lf-reg-inc",
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

        df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "taraban_item_df.csv")
        )

        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
            .mean()
            .reset_index()
        )

        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "taraban_mean_df.csv")
        )

        self.taraban_mean_df = mean_df
        
        return df

    def _eval_cortese(self):

        df = pd.DataFrame()
        for testset_name in ("cortese_hi_img", "cortese_low_img"):
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

        df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "cortese_item_df.csv")
        )

        mean_df = (
            df.groupby(["code_name", "task", "testset", "epoch", "timetick", "y"])
            .mean()
            .reset_index()
        )

        mean_df.to_csv(
            os.path.join(self.cfg.path["model_folder"], "eval", "cortese_mean_df.csv")
        )

        self.cortese_mean_df = mean_df
        
        return df

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