# This file need serious refactoring.


import pandas as pd
from dotenv import load_dotenv

import evaluate
import plot as p
from meta import Config

load_dotenv()


def run_oral_eval(cfg: Config, testset: str = "train_r100") -> None:
    """Runs the oral evaluation and plot the accuracy and SSE in oral tasks."""
    t = evaluate.Test(cfg)
    oral_tasks = ["pho_pho", "sem_sem", "sem_pho", "pho_sem"]
    results = [make_mean_df(t.eval(testset, task)) for task in oral_tasks]
    df = pd.concat(results, ignore_index=True)

    p.plot_metric_over_epoch(
        df, metric="acc", save=f"{cfg.plot_folder}/oral_acc_{testset}.html"
    )
    p.plot_metric_over_epoch(
        df, metric="sse", save=f"{cfg.plot_folder}/oral_sse_{testset}.html"
    )
    p.plot_metric_over_epoch(
        df, metric="csse", save=f"{cfg.plot_folder}/oral_csse_{testset}.html"
    )


def run_oral_homophone(cfg: Config) -> None:
    t = evaluate.Test(cfg)
    tasks = ["sem_pho", "pho_sem"]
    results = [make_cond_mean_df(t.eval("homophony", task)) for task in tasks]
    df = pd.concat(results, ignore_index=True)
    p.plot_homophony(df, save=f"{cfg.plot_folder}/oral_homophony.html")


def run_read_eval(cfg: Config, testset: str = "train_r100") -> None:
    """Runs the read evaluation and plot the accuracy and SSE in read tasks."""
    t = evaluate.Test(cfg)
    df = t.eval(testset, "triangle")
    p.plot_triangle(
        df, metric="acc", save=f"{cfg.plot_folder}/triangle_acc_{testset}.html"
    )


def run_lesion(cfg: Config, testset: str = "train_r100"):
    """Leison in Semantic (HS04 fig.14)."""
    import metrics

    t = evaluate.Test(cfg)

    # Override semantic accuracy with cosine accuracy
    t.metrics["acc"]["sem"] = metrics.CosineSemanticAccuracy()

    # SEM (same as HS04 but with Cosine)
    df_intact = t.eval(testset, "triangle", save_file_prefix="cos")
    df_os_lesion = t.eval(testset, "exp_ops", save_file_prefix="cos")
    df_ops_lesion = t.eval(testset, "ort_sem", save_file_prefix="cos")

    df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion], ignore_index=True)
    mdf_sem = make_mean_df(df_sem)
    p.plot_metric_over_epoch(
        mdf_sem, output="sem", save=f"{cfg.plot_folder}/lesion_SEM_{testset}.html"
    )


def run_lesion_extra(cfg: Config, testset: str = "train_r100"):
    """Lesion in Phonology."""
    t = evaluate.Test(cfg)
    df_intact = t.eval(testset, "triangle")
    df_op_lesion = t.eval(testset, "exp_osp")
    df_osp_lesion = t.eval(testset, "ort_pho")

    df_pho = pd.concat([df_intact, df_op_lesion, df_osp_lesion], ignore_index=True)
    mdf_pho = make_mean_df(df_pho)
    p.plot_metric_over_epoch(
        mdf_pho, output="pho", save=f"{cfg.plot_folder}/lesion_PHO_{testset}.html"
    )


# def run_test2(code_name, batch_name=None, task="triangle"):
#     test = init(code_name, batch_name)
#     df = test.eval("taraban", task)
#     mdf = make_cond_mean_df(df)

#     mdf = mdf.loc[
#         mdf.cond.isin(
#             [
#                 "High-frequency exception",
#                 "Regular control for High-frequency exception",
#                 "Low-frequency exception",
#                 "Regular control for Low-frequency exception",
#             ]
#         )
#     ]
#     mdf["freq"] = mdf.cond.apply(
#         lambda x: "High"
#         if x
#         in ("High-frequency exception", "Regular control for High-frequency exception")
#         else "Low"
#     )
#     mdf["reg"] = mdf.cond.apply(
#         lambda x: "Regular" if x.startswith("Regular") else "Exception"
#     )

#     fig10_acc = plot_hs04_fig10(mdf, metric="acc")
#     fig10_acc.save(os.path.join(test.cfg.plot_folder, f"test2_acc_{task}.html"))

#     fig10_sse = plot_hs04_fig10(mdf, metric="csse")
#     fig10_sse.save(os.path.join(test.cfg.plot_folder, f"test2_sse_{task}.html"))


# def run_test3(code_name, batch_name=None):
#     test = init(code_name, batch_name)
#     df = test.eval("glushko", "triangle")
#     mdf = make_cond_mean_df(df)
#     test3_acc = plot_conds(mdf, "acc")
#     test3_acc.save(os.path.join(test.cfg.plot_folder, "test3_acc.html"))

#     test3_sse = plot_conds(mdf, "csse")
#     test3_sse.save(os.path.join(test.cfg.plot_folder, "test3_sse.html"))


# def run_test4(code_name, batch_name=None):
#     test = init(code_name, batch_name)
#     df = test.eval("hs04_img_240", "triangle")
#     mdf = make_cond_mean_df(df)
#     mdf["fc"] = mdf.cond.apply(lambda x: x[:5])
#     mdf["img"] = mdf.cond.apply(lambda x: x[-2:])
#     test4_acc = plot_hs04_fig11(mdf, metric="acc")
#     test4_acc.save(os.path.join(test.cfg.plot_folder, "test4_acc.html"))

#     test4_sse = plot_hs04_fig11(mdf, metric="csse")
#     test4_sse.save(os.path.join(test.cfg.plot_folder, "test4_sse.html"))


# def run_test5(code_name, batch_name=None):
#     tau = 1.0 / 12.0
#     test = init(code_name, batch_name, tau_override=tau)

#     # SEM (same as HS04)
#     df_intact = test.eval("train_r100", "triangle", save_file_prefix="hi_res")
#     df_os_lesion = test.eval("train_r100", "exp_ops", save_file_prefix="hi_res")
#     df_ops_lesion = test.eval("train_r100", "ort_sem", save_file_prefix="hi_res")

#     df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion], ignore_index=True)
#     mdf_sem = make_mean_df(df_sem)
#     test5a = plot_hs04_fig14(mdf_sem, output="sem")
#     test5a.save(os.path.join(test.cfg.plot_folder, "test5_sem.html"))

#     # PHO (extra)
#     df_op_lesion = test.eval("train_r100", "exp_osp", save_file_prefix="hi_res")
#     df_osp_lesion = test.eval("train_r100", "ort_pho", save_file_prefix="hi_res")

#     df_pho = pd.concat([df_intact, df_op_lesion, df_osp_lesion], ignore_index=True)
#     mdf_pho = make_mean_df(df_pho)

#     test5b = plot_hs04_fig14(mdf_pho, output="pho")
#     test5b.save(os.path.join(test.cfg.plot_folder, "test5_pho.html"))


# def run_test6_cosine_split(code_name, batch_name=None, testset="train_r100"):
#     """Test 6 using cosine accuracy in SEM and median split by cond"""

#     test = init(code_name, batch_name)
#     test.cfg.tf_root = "/home/jupyter/triangle_model"
#     import metrics

#     # Override semantic accuracy with cosine accuracy
#     test.METRICS_MAP["acc"]["sem"] = metrics.CosineSemanticAccuracy()

#     # Override tf_root
#     # test.cfg.tf_root = os.path.join("/home/jupyter/triangle_model")

#     # SEM (same as HS04)
#     df_intact = test.eval(testset, "triangle", save_file_prefix="cos")
#     df_os_lesion = test.eval(testset, "exp_ops", save_file_prefix="cos")
#     df_ops_lesion = test.eval(testset, "ort_sem", save_file_prefix="cos")

#     df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion], ignore_index=True)

#     df_sem_hf = df_sem.loc[df_sem.cond == "hf"].copy()
#     df_sem_lf = df_sem.loc[df_sem.cond == "lf"].copy()

#     mdf_sem_hf = make_mean_df(df_sem_hf)
#     mdf_sem_lf = make_mean_df(df_sem_lf)

#     plot_hs04_fig14(mdf_sem_hf, output="sem").save(
#         os.path.join(test.cfg.plot_folder, f"test6_sem_cosine_hf_{testset}.html")
#         )

#     plot_hs04_fig14(mdf_sem_lf, output="sem").save(
#         os.path.join(test.cfg.plot_folder, f"test6_sem_cosine_lf_{testset}.html")
#         )

# def run_test6r(code_name, batch_name=None, testset="train_r300_difficulty"):
#     """Lesioning with high_mid_low difficulty"""
#     test = init(code_name, batch_name)

#     # Get word to condition mapping
#     ts = data_wrangling.load_testset(testset)
#     word_to_cond = dict(zip(ts["item"], ts["cond"]))

#     # SEM (same as HS04)
#     df_intact = test.eval(testset, "triangle")
#     df_os_lesion = test.eval(testset, "exp_ops")
#     df_ops_lesion = test.eval(testset, "ort_sem")

#     df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion], ignore_index=True)
#     df_sem["cond"] = df_sem["word"].map(word_to_cond)
#     mdf_sem = make_cond_mean_df(df_sem)

#     for diff in ["hi", "mid", "low"]:
#         test5a = plot_hs04_fig14(mdf_sem.loc[mdf_sem.cond == diff], output="sem")
#         test5a.save(
#             os.path.join(test.cfg.plot_folder, f"test6r_sem_{diff}_{testset}.html")
#         )

#     # PHO (extra)
#     df_op_lesion = test.eval(testset, "exp_osp")
#     df_osp_lesion = test.eval(testset, "ort_pho")

#     df_pho = pd.concat([df_intact, df_op_lesion, df_osp_lesion], ignore_index=True)
#     df_pho["cond"] = df_pho["word"].map(word_to_cond)
#     mdf_pho = make_cond_mean_df(df_pho)

#     for diff in ["hi", "mid", "low"]:
#         test5b = plot_hs04_fig14(mdf_pho.loc[mdf_pho.cond == diff], output="pho")
#         test5b.save(
#             os.path.join(test.cfg.plot_folder, f"test6r_pho_{diff}_{testset}.html")
#         )


# def run_test7(code_name, batch_name=None):
#     """Activation by target signal"""
#     test = init(code_name, batch_name)
#     # SEM (same as HS04)
#     df1_sem = test.eval("train_r100", "triangle")
#     df2_sem = test.eval("train_r100", "exp_ops")
#     df3_sem = test.eval("train_r100", "ort_sem")
#     df_sem = pd.concat([df1_sem, df2_sem, df3_sem], ignore_index=True)
#     p_sem = plot_activation_by_target(df_sem, output="sem")
#     p_sem.save(os.path.join(test.cfg.plot_folder, "test7_sem.html"))

#     # PHO
#     df1_pho = test.eval("train_r100", "triangle")
#     df2_pho = test.eval("train_r100", "exp_osp")
#     df3_pho = test.eval("train_r100", "ort_pho")

#     df_pho = pd.concat([df1_pho, df2_pho, df3_pho], ignore_index=True)
#     p_pho = plot_activation_by_target(df_pho, output="pho")
#     p_pho.save(os.path.join(test.cfg.plot_folder, "test7_pho.html"))


# def run_test8(code_name, batch_name=None, epoch=None):
#     """10 random word raw input temporal dynamics"""
#     d = examine.Diagnosis(code_name)

#     if epoch is None:
#         # Use last epoch if no epoch is provided
#         epoch = d.cfg.total_number_of_epoch

#     d.eval("train_r100", task="triangle", epoch=epoch)

#     ten_words = random.sample(d.testset_package["item"], 10)

#     ten_plots_pho = alt.hconcat()
#     ten_plots_sem = alt.hconcat()
#     for w in ten_words:
#         ten_plots_pho &= plot_raw_input_by_target(w, d, "pho")
#         ten_plots_sem &= plot_raw_input_by_target(w, d, "sem")

#     ten_plots_pho.resolve_scale(y="shared").save(
#         os.path.join(d.cfg.plot_folder, "test8_pho.html")
#     )
#     ten_plots_sem.resolve_scale(y="shared").save(
#         os.path.join(d.cfg.plot_folder, "test8_sem.html")
#     )


# def run_test9(code_name, batch_name=None, epoch=None):
#     """Compare weight with mikenet"""
#     d = examine.Diagnosis(code_name)

#     if epoch is None:
#         # Use last epoch if no epoch is provided
#         epoch = d.cfg.total_number_of_epoch

#     d.eval("train_r100", task="triangle", epoch=epoch)

#     w = examine.MikeNetWeight("mikenet/Reading_Weight_v1")
#     os.makedirs(os.path.join(d.cfg.plot_folder, "compare_mn_weights"), exist_ok=True)

#     for x in w.weight_keys:
#         w_name = w.as_tf_name(x)

#         try:
#             tmp = examine.dual_plot(d, w, w_name)
#             tmp.savefig(
#                 os.path.join(d.cfg.plot_folder, "compare_mn_weights", f"{w_name}.png")
#             )
#         except IndexError:
#             pass


# def run_test10(code_name, batch_name=None):
#     """Test 10: Evaluate the accuracy in each task over epoch with 3 levels of difficulties"""

#     test = init(code_name, batch_name)

#     testset = "train_r300_difficulty"
#     ts = data_wrangling.load_testset(testset)

#     tasks = ["triangle", "pho_pho", "sem_sem", "sem_pho", "pho_sem"]

#     # Evaluate all tasks
#     all_eval = [test.eval(testset, x) for x in tasks]
#     df = pd.concat(all_eval, ignore_index=True)

#     # Post process the data
#     word_to_cond = dict(zip(ts["item"], ts["cond"]))
#     df["cond"] = df["word"].map(word_to_cond)
#     mdf = make_cond_mean_df(df)

#     # Plot and save
#     plot_acc_by_difficulty(mdf).save(os.path.join(test.cfg.plot_folder, "test10.html"))


# ########## Support functions ##########


# def plot_raw_input_by_target(
#     word: str, diag: examine.Diagnosis, layer: str
# ) -> alt.Chart:
#     diag.set_target_word(word)
#     print(f"Output phoneme over timeticks: {diag.list_output_phoneme}")

#     df_act1 = diag.subset_df(layer=layer, target_act=1)
#     plotter = examine.Plots(df_act1)
#     raw_input_1 = plotter.raw_inputs()

#     df_act0 = diag.subset_df(layer=layer, target_act=0)
#     plotter = examine.Plots(df_act0)
#     raw_input_0 = plotter.raw_inputs()

#     return (raw_input_1 | raw_input_0).resolve_scale(y="shared").properties(title=word)


# def print_unique(df):
#     for x in ("testset", "task", "output_name", "timetick", "cond"):
#         print(f"{x}: unique entry: {df[x].unique()}")


def make_mean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Averaging on items axis."""
    df["csse"] = df.sse.loc[df.acc == 1]
    gp_vars = ["code_name", "epoch", "testset", "task", "output_name", "timetick"]
    return df.groupby(gp_vars).mean().reset_index()


def make_cond_mean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate on items axis with condition."""
    df["csse"] = df.sse.loc[df.acc == 1]
    gp_vars = [
        "code_name",
        "epoch",
        "testset",
        "task",
        "output_name",
        "timetick",
        "cond",
    ]
    return df.groupby(gp_vars).mean().reset_index()


# def plot_hs04_fig9(mean_df, metric="acc"):
#     """test case 1"""

#     interval = alt.selection_interval()
#     metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

#     timetick_sel = (
#         alt.Chart(mean_df)
#         .mark_rect()
#         .encode(x="timetick:O", color=f"mean({metric}):Q")
#         .add_selection(interval)
#     ).properties(width=400)

#     main = (
#         alt.Chart(mean_df)
#         .mark_line(point=True)
#         .encode(
#             x="epoch:Q",
#             y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
#             color="output_name:N",
#             tooltip=[f"mean({metric})"]
#         )
#         .interactive()
#         .transform_filter(interval)
#     )

#     return timetick_sel & main


# def plot_hs04_fig10(mean_df, metric="acc"):
#     """test case 2"""
#     metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()
#     mean_df = mean_df.loc[mean_df.output_name == "pho"]

#     interval_epoch = alt.selection_interval(init={"epoch": (305, 315)})
#     interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})

#     epoch_selection = (
#         alt.Chart(mean_df).mark_rect().encode(x="epoch:Q").add_selection(interval_epoch)
#     ).properties(width=400)

#     timetick_sel = (
#         alt.Chart(mean_df)
#         .mark_rect()
#         .encode(x="timetick:Q")
#         .add_selection(interval_timetick)
#     ).properties(width=400)

#     plot_fxc_interact = (
#         alt.Chart(mean_df)
#         .mark_line()
#         .encode(
#             x=alt.X("freq:N", scale=alt.Scale(reverse=True)),
#             y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
#             color="reg:N",
#         )
#         .transform_filter(interval_timetick)
#         .transform_filter(interval_epoch)
#         .properties(width=400)
#     )

#     return epoch_selection & timetick_sel & plot_fxc_interact


# def plot_conds(mean_df, metric="acc"):
#     """test case 3"""
#     metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

#     mean_df = mean_df.loc[mean_df.output_name == "pho"]

#     interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})
#     timetick_sel = (
#         alt.Chart(mean_df)
#         .mark_rect()
#         .encode(x="timetick:Q")
#         .add_selection(interval_timetick)
#     ).properties(width=400)

#     cond_over_epoch = (
#         alt.Chart(mean_df)
#         .mark_line()
#         .encode(
#             x="epoch:Q",
#             y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
#             color="cond:N",
#         )
#         .transform_filter(interval_timetick)
#         .interactive()
#     )

#     return timetick_sel & cond_over_epoch


# def plot_hs04_fig11(mean_df, metric="acc"):
#     """test case 4"""
#     mean_df = mean_df.loc[mean_df.output_name == "pho"]
#     metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

#     interval_epoch = alt.selection_interval(init={"epoch": (305, 315)})
#     interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})

#     epoch_selection = (
#         alt.Chart(mean_df).mark_rect().encode(x="epoch:Q").add_selection(interval_epoch)
#     ).properties(width=400)

#     timetick_sel = (
#         alt.Chart(mean_df)
#         .mark_rect()
#         .encode(x="timetick:Q")
#         .add_selection(interval_timetick)
#     ).properties(width=400)

#     bar = (
#         alt.Chart(mean_df)
#         .mark_bar()
#         .encode(
#             x="img:N",
#             y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
#             color="img:N",
#             column="fc:N",
#         )
#         .transform_filter(interval_epoch)
#         .transform_filter(interval_timetick)
#         .properties(width=50, height=200)
#     )
#     return epoch_selection & timetick_sel & bar


# def plot_activation_by_target(df, output):
#     df = df.loc[df.output_name == output]

#     interval_epoch = alt.selection_interval(init={"epoch": (305, 315)})
#     interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})

#     epoch_selection = (
#         alt.Chart(df).mark_rect().encode(x="epoch:Q").add_selection(interval_epoch)
#     ).properties(width=400)

#     timetick_sel = (
#         alt.Chart(df)
#         .mark_rect()
#         .encode(x="timetick:Q")
#         .add_selection(interval_timetick)
#     ).properties(width=400)

#     plot_activation = (
#         alt.Chart(df)
#         .mark_point()
#         .encode(
#             x=alt.X(
#                 "act1:Q",
#                 scale=alt.Scale(domain=(0, 1)),
#                 title="Activation for target node = 1",
#             ),
#             y=alt.Y(
#                 "act0:Q",
#                 scale=alt.Scale(reverse=True, domain=(0, 1)),
#                 title="Activation for target node = 0",
#             ),
#             color="task:N",
#             detail="word:N",
#             column=alt.Column("acc:N", title="Prediction accuracy"),
#             tooltip=["word", "acc", "sse", "act0", "act1"],
#         )
#         .transform_filter(interval_epoch)
#         .transform_filter(interval_timetick)
#         .properties(title=f"{output} diagnostic")
#     )

#     return epoch_selection & timetick_sel & plot_activation


# def plot_acc_by_difficulty(mean_df, metric="acc"):
#     """A quick and dirty plot to check the impact of item difficulty on task performance"""

#     interval = alt.selection_interval()
#     metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

#     timetick_sel = (
#         alt.Chart(mean_df)
#         .mark_rect()
#         .encode(x="timetick:O", color=f"mean({metric}):Q")
#         .add_selection(interval)
#     ).properties(width=400)

#     main = (
#         alt.Chart(mean_df)
#         .mark_line(point=True)
#         .encode(
#             x="epoch:Q",
#             y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
#             color=alt.Color("cond:O", sort=["hi", "mid", "low"], title="difficulty"),
#             column="output_name:N",
#             row="task:N",
#         )
#         .transform_filter(interval)
#     )

#     return timetick_sel & main


# ################################################################################

# TEST_MAP = {
#     1: run_test1,
#     2: run_test2,
#     3: run_test3,
#     4: run_test4,
#     5: run_test5,
#     6: run_test6,
#     7: run_test7,
#     8: run_test8,
#     9: run_test9,
#     10: run_test10,
#     11: run_test6r,
#     12: run_test6_cosine,
# }


# def main(code_name: str, batch_name: str=None):
#     """Run the frequently used tests (except high res test 5)"""
#     for i in (1, 2, 3, 4, 6, 12):
#         try:
#             TEST_MAP[i](code_name, batch_name)
#         except Exception:
#             print(f"Test {i} failed")
#             pass


# if __name__ == "__main__":
#     """Command line entry point, take code_name and testcase to run tests"""
#     parser = argparse.ArgumentParser(description="Run HS04 test cases")
#     parser.add_argument("-n", "--code_name", required=True, type=str)
#     parser.add_argument("-b", "--batch_name", type=str)
#     args = parser.parse_args()
#     main(args.code_name, args.batch_name)
