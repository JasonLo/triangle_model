import argparse, os
import pandas as pd
import altair as alt
import meta, evaluate, troubleshooting
import modeling
import random



def init(code_name, tau_override=None):

    cfg_json = os.path.join("models", code_name, "model_config.json")
    cfg = meta.ModelConfig.from_json(cfg_json)

    # Force output to 13:
    cfg.output_ticks = 13

    # Rebuild model with tau_override
    if tau_override is not None:
        cfg.tau_original = cfg.tau
        cfg.tau = tau_override
        cfg.output_ticks = int(round(cfg.output_ticks * (cfg.tau_original / cfg.tau)))

    test = evaluate.TestSet(cfg)
    return test


def run_test1(code_name):
    test = init(code_name)
    df = test.eval("train_r100", "triangle")
    mdf = make_mean_df(df)
    fig9 = plot_hs04_fig9(mdf, metric="acc")
    fig9_sse = plot_hs04_fig9(mdf, metric="csse")
    fig9.save(os.path.join(test.cfg.plot_folder, "test1_acc.html"))
    fig9_sse.save(os.path.join(test.cfg.plot_folder, "test1_sse.html"))

    # Extras Oral tasks
    mdf_pp = make_mean_df(test.eval("train_r100", "pho_pho"))
    mdf_ss = make_mean_df(test.eval("train_r100", "sem_sem"))
    mdf_sp = make_mean_df(test.eval("train_r100", "sem_pho"))
    mdf_ps = make_mean_df(test.eval("train_r100", "pho_sem"))

    df_oral = pd.concat([mdf_pp, mdf_ss, mdf_sp, mdf_ps], ignore_index=True)
    test1_oral_plot_acc = plot_hs04_fig14(df_oral, metric="acc")
    test1_oral_plot_sse = plot_hs04_fig14(df_oral, metric="csse")

    test1_oral_plot_acc.save(os.path.join(test.cfg.plot_folder, "test1_oral_acc.html"))
    test1_oral_plot_sse.save(os.path.join(test.cfg.plot_folder, "test1_oral_sse.html"))


def run_test2(code_name):
    test = init(code_name)
    df = test.eval("taraban", "triangle")
    mdf = make_cond_mean_df(df)

    mdf = mdf.loc[
        mdf.cond.isin(
            [
                "High-frequency exception",
                "Regular control for High-frequency exception",
                "Low-frequency exception",
                "Regular control for Low-frequency exception",
            ]
        )
    ]
    mdf["freq"] = mdf.cond.apply(
        lambda x: "High"
        if x
        in ("High-frequency exception", "Regular control for High-frequency exception")
        else "Low"
    )
    mdf["reg"] = mdf.cond.apply(
        lambda x: "Regular" if x.startswith("Regular") else "Exception"
    )

    fig10_acc = plot_hs04_fig10(mdf, metric="acc")
    fig10_acc.save(os.path.join(test.cfg.plot_folder, "test2_acc.html"))

    fig10_sse = plot_hs04_fig10(mdf, metric="csse")
    fig10_sse.save(os.path.join(test.cfg.plot_folder, "test2_sse.html"))


def run_test3(code_name):
    test = init(code_name)
    df = test.eval("glushko", "triangle")
    mdf = make_cond_mean_df(df)
    test3_acc = plot_conds(mdf, "acc")
    test3_acc.save(os.path.join(test.cfg.plot_folder, "test3_acc.html"))

    test3_sse = plot_conds(mdf, "csse")
    test3_sse.save(os.path.join(test.cfg.plot_folder, "test3_sse.html"))

def run_test4(code_name):
    test = init(code_name)
    df = test.eval("hs04_img_240", "triangle")
    mdf = make_cond_mean_df(df)
    mdf["fc"] = mdf.cond.apply(lambda x: x[:5])
    mdf["img"] = mdf.cond.apply(lambda x: x[-2:])
    test4_acc = plot_hs04_fig11(mdf, metric="acc")
    test4_acc.save(os.path.join(test.cfg.plot_folder, "test4_acc.html"))

    test4_sse = plot_hs04_fig11(mdf, metric="csse")
    test4_sse.save(os.path.join(test.cfg.plot_folder, "test4_sse.html"))

def run_test5(code_name):
    tau = 1.0 / 12.0
    test = init(code_name, tau_override=tau)

    # SEM (same as HS04)
    df_intact = test.eval("train_r100", "triangle", save_file_prefix="hi_res")
    df_os_lesion = test.eval("train_r100", "exp_ops", save_file_prefix="hi_res")
    df_ops_lesion = test.eval("train_r100", "ort_sem", save_file_prefix="hi_res")

    df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion], ignore_index=True)
    mdf_sem = make_mean_df(df_sem)
    test5a = plot_hs04_fig14(mdf_sem, output="sem")
    test5a.save(os.path.join(test.cfg.plot_folder, "test5_sem.html"))

    # PHO (extra)
    df_op_lesion = test.eval("train_r100", "exp_osp", save_file_prefix="hi_res")
    df_osp_lesion = test.eval("train_r100", "ort_pho", save_file_prefix="hi_res")

    df_pho = pd.concat([df_intact, df_op_lesion, df_osp_lesion], ignore_index=True)
    mdf_pho = make_mean_df(df_pho)

    test5b = plot_hs04_fig14(mdf_pho, output="pho")
    test5b.save(os.path.join(test.cfg.plot_folder, "test5_pho.html"))

def run_test6(code_name):
    """Test 5 without resizing tau"""
    test = init(code_name)

    # SEM (same as HS04)
    df_intact = test.eval("train_r100", "triangle")
    df_os_lesion = test.eval("train_r100", "exp_ops")
    df_ops_lesion = test.eval("train_r100", "ort_sem")

    df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion], ignore_index=True)
    mdf_sem = make_mean_df(df_sem)
    test5a = plot_hs04_fig14(mdf_sem, output="sem")
    test5a.save(os.path.join(test.cfg.plot_folder, "test6_sem.html"))

    # PHO (extra)
    df_op_lesion = test.eval("train_r100", "exp_osp")
    df_osp_lesion = test.eval("train_r100", "ort_pho")

    df_pho = pd.concat([df_intact, df_op_lesion, df_osp_lesion], ignore_index=True)
    mdf_pho = make_mean_df(df_pho)

    test5b = plot_hs04_fig14(mdf_pho, output="pho")
    test5b.save(os.path.join(test.cfg.plot_folder, "test6_pho.html"))


def run_test7(code_name):
    """Activation by target signal"""
    test = init(code_name)
    # SEM (same as HS04)
    df1_sem = test.eval("train_r100", "triangle")
    df2_sem = test.eval("train_r100", "exp_ops")
    df3_sem = test.eval("train_r100", "ort_sem")
    df_sem = pd.concat([df1_sem, df2_sem, df3_sem], ignore_index=True)
    p_sem = plot_activation_by_target(df_sem, output="sem")
    p_sem.save(os.path.join(test.cfg.plot_folder, "test7_sem.html"))

    # PHO
    df1_pho = test.eval("train_r100", "triangle")
    df2_pho = test.eval("train_r100", "exp_osp")
    df3_pho = test.eval("train_r100", "ort_pho")

    df_pho = pd.concat([df1_pho, df2_pho, df3_pho], ignore_index=True)
    p_pho = plot_activation_by_target(df_pho, output="pho")
    p_pho.save(os.path.join(test.cfg.plot_folder, "test7_pho.html"))

def run_test8(code_name, epoch=None):
    """10 random word raw input temporal dynamics"""
    d = troubleshooting.Diagnosis(code_name)

    if epoch is None:
        # Use last epoch if no epoch is provided
        epoch = d.cfg.total_number_of_epoch

    d.eval('train_r100', task='triangle', epoch=epoch)

    ten_words = random.sample(d.testset_package['item'], 10)

    ten_plots_pho = alt.hconcat()
    ten_plots_sem = alt.hconcat()
    for w in ten_words:
        ten_plots_pho &= plot_raw_input_by_target(w, d, 'pho')
        ten_plots_sem &= plot_raw_input_by_target(w, d, 'sem')

    ten_plots_pho.resolve_scale(y="shared").save(os.path.join(d.cfg.plot_folder, 'test8_pho.html'))
    ten_plots_sem.resolve_scale(y="shared").save(os.path.join(d.cfg.plot_folder, 'test8_sem.html'))


def run_test9(code_name, epoch=None):
    """ Compare weight with mikenet """
    d = troubleshooting.Diagnosis(code_name)

    if epoch is None:
        # Use last epoch if no epoch is provided
        epoch = d.cfg.total_number_of_epoch

    d.eval('train_r100', task='triangle', epoch=epoch)

    w = troubleshooting.MikeNetWeight('mikenet/Reading_Weight_v1')
    os.makedirs(os.path.join(d.cfg.plot_folder, "compare_mn_weights"), exist_ok=True)

    for x in w.weight_keys:
        w_name = w.as_tf_name(x)

        try:
            tmp = troubleshooting.dual_plot(d, w, w_name)
            tmp.savefig(os.path.join(d.cfg.plot_folder, "compare_mn_weights", f"{w_name}.png"))
        except IndexError:
            pass
    



########## Support functions ##########

def plot_raw_input_by_target(word:str, diag:troubleshooting.Diagnosis, layer:str) -> alt.Chart:
    diag.set_target_word(word)
    print(f"Output phoneme over timeticks: {diag.list_output_phoneme}")

    df_act1 = diag.subset_df(layer=layer, target_act=1)
    plotter = troubleshooting.Plots(df_act1)
    raw_input_1 = plotter.raw_inputs()

    df_act0 = diag.subset_df(layer=layer, target_act=0)
    plotter = troubleshooting.Plots(df_act0)
    raw_input_0 = plotter.raw_inputs()

    return (raw_input_1 | raw_input_0).resolve_scale(y="shared").properties(title=word)


def print_unique(df):
    for x in ("testset", "task", "output_name", "timetick", "cond"):
        print(f"{x}: unique entry: {df[x].unique()}")


def make_mean_df(df):
    """Aggregate on items axis to one value"""
    df["csse"] = df.sse.loc[df.acc == 1]
    gp_vars = ["code_name", "epoch", "testset", "task", "output_name", "timetick"]
    return df.groupby(gp_vars).mean().reset_index()


def make_cond_mean_df(df):
    """Aggregate on items axis with condition"""
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


def plot_hs04_fig9(mean_df, metric="acc"):
    """test case 1"""

    interval = alt.selection_interval()
    metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

    timetick_sel = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="timetick:O", color=f"mean({metric}):Q")
        .add_selection(interval)
    ).properties(width=400)

    main = (
        alt.Chart(mean_df)
        .mark_line(point=True)
        .encode(
            x="epoch:Q",
            y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
            color="output_name:N",
        )
        .transform_filter(interval)
    )

    return timetick_sel & main



def plot_hs04_fig10(mean_df, metric="acc"):
    """test case 2"""
    metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()
    mean_df = mean_df.loc[mean_df.output_name == "pho"]

    interval_epoch = alt.selection_interval(init={"epoch": (305, 315)})
    interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})


    epoch_selection = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="epoch:Q")
        .add_selection(interval_epoch)
    ).properties(width=400)

    timetick_sel = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="timetick:Q")
        .add_selection(interval_timetick)
    ).properties(width=400)

    plot_fxc_interact = (alt.Chart(mean_df)
        .mark_line()
        .encode(
            x=alt.X("freq:N", scale=alt.Scale(reverse=True)),
            y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
            color="reg:N",
        )
        .transform_filter(interval_timetick)
        .transform_filter(interval_epoch)
        .properties(width=400)
    )
    

    return epoch_selection & timetick_sel & plot_fxc_interact
        


def plot_conds(mean_df, metric="acc"):
    """test case 3"""
    metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

    mean_df = mean_df.loc[mean_df.output_name == "pho"]

    interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})
    timetick_sel = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="timetick:Q")
        .add_selection(interval_timetick)
    ).properties(width=400)

    cond_over_epoch = (
        alt.Chart(mean_df)
        .mark_line()
        .encode(
            x="epoch:Q",
            y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
            color="cond:N",
        )
        .transform_filter(interval_timetick)
        .interactive()
    )
    

    return timetick_sel & cond_over_epoch


def plot_hs04_fig11(mean_df, metric="acc"):
    """test case 4"""
    mean_df = mean_df.loc[mean_df.output_name == "pho"]
    metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()


    interval_epoch = alt.selection_interval(init={"epoch": (305, 315)})
    interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})

    epoch_selection = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="epoch:Q")
        .add_selection(interval_epoch)
    ).properties(width=400)

    timetick_sel = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="timetick:Q")
        .add_selection(interval_timetick)
    ).properties(width=400)

    bar = (
        alt.Chart(mean_df)
        .mark_bar()
        .encode(x="img:N", y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale), color="img:N", column="fc:N")
        .transform_filter(interval_epoch)
        .transform_filter(interval_timetick)
        .properties(width=50, height=200)
    )
    return epoch_selection & timetick_sel & bar


def plot_hs04_fig14(mean_df, output=None, metric="acc"):

    if output is not None:
        mean_df = mean_df.loc[(mean_df.output_name == output)]

    metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

    interval = alt.selection_interval()

    timetick_sel = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="timetick:O", color=f"mean({metric}):Q")
        .add_selection(interval)
    ).properties(width=400)

    line = (
        alt.Chart(mean_df)
        .mark_line(point=True)
        .encode(
            x="epoch:Q",
            y=alt.Y(f"mean({metric}):Q", scale=metric_specific_scale),
            color="task:N",
            column="output_name:N",
        )
        .transform_filter(interval)
    )

    return timetick_sel & line

def plot_activation_by_target(df, output):
    df = df.loc[df.output_name==output]

    interval_epoch = alt.selection_interval(init={"epoch": (305, 315)})
    interval_timetick = alt.selection_interval(init={"timetick": (4, 12)})

    epoch_selection = (
        alt.Chart(df)
        .mark_rect()
        .encode(x="epoch:Q")
        .add_selection(interval_epoch)
    ).properties(width=400)

    timetick_sel = (
        alt.Chart(df)
        .mark_rect()
        .encode(x="timetick:Q")
        .add_selection(interval_timetick)
    ).properties(width=400)

    plot_activation = (
        alt.Chart(df).mark_point().encode(
        x=alt.X("act1:Q", scale=alt.Scale(domain=(0,1)), title="Activation for target node = 1"),
        y=alt.Y("act0:Q", scale=alt.Scale(reverse=True, domain=(0,1)), title="Activation for target node = 0"),
        color="task:N",
        detail="word:N",
        column=alt.Column("acc:N", title="Prediction accuracy"),
        tooltip=['word', 'acc', 'sse', 'act0', 'act1' ]
        )
        .transform_filter(interval_epoch)
        .transform_filter(interval_timetick)
        .properties(title=f"{output} diagnostic")
    )

    return epoch_selection & timetick_sel & plot_activation

################################################################################

TEST_MAP = {
    1: run_test1, 2: run_test2, 3: run_test3, 
    4: run_test4, 5: run_test5, 6: run_test6,
    7: run_test7, 8: run_test8
}

def main(code_name):
    """Run the frequently used tests"""
    for i in (1, 2, 3, 4, 6, 7, 8):
        TEST_MAP[i](code_name)


if __name__ == "__main__":
    """Command line entry point, take code_name and testcase to run tests"""
    parser = argparse.ArgumentParser(description="Run HS04 test cases")
    parser.add_argument("-n", "--code_name", required=True, type=str)
    args = parser.parse_args()
    main(args.code_name)
