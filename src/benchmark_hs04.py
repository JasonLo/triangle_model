import argparse, os
import pandas as pd
import altair as alt
import meta, modeling, evaluate


def init(code_name, tau_override=None):

    cfg_json = os.path.join("models", code_name, "model_config.json")
    cfg = meta.ModelConfig.from_json(cfg_json)

    # Rebuild model with tau_override
    if tau_override is not None:
        cfg.tau_original = cfg.tau
        cfg.tau = tau_override
        cfg.output_ticks = int(round(cfg.output_ticks * (cfg.tau_original / cfg.tau)))

    model = modeling.MyModel(cfg)
    test = evaluate.TestSet(cfg, model)
    return test


def run_test1(code_name):
    test = init(code_name)
    df = test.eval("train_r1000", "triangle")
    mdf = make_mean_df(df)
    fig9 = plot_hs04_fig9(mdf, metric="acc")
    fig9_sse = plot_hs04_fig9(mdf, metric="csse")
    fig9.save(os.path.join(test.cfg.plot_folder, "test1_acc.html"))
    fig9_sse.save(os.path.join(test.cfg.plot_folder, "test1_sse.html"))

    # Extras Oral tasks
    mdf_pp = make_mean_df(test.eval("train_r1000", "pho_pho"))
    mdf_ss = make_mean_df(test.eval("train_r1000", "sem_sem"))
    mdf_sp = make_mean_df(test.eval("train_r1000", "sem_pho"))
    mdf_ps = make_mean_df(test.eval("train_r1000", "pho_sem"))

    df_oral = pd.concat([mdf_pp, mdf_ss, mdf_sp, mdf_ps])
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

    fig10 = plot_hs04_fig10(
        mdf, max_epoch=test.cfg.total_number_of_epoch, tick_after=12
    )
    fig10.save(os.path.join(test.cfg.plot_folder, "test2.html"))


def run_test3(code_name):
    test = init(code_name)
    df = test.eval("glushko", "triangle")
    mdf = make_cond_mean_df(df)
    test3 = plot_conds(mdf, tick_after=12)
    test3.save(os.path.join(test.cfg.plot_folder, "test3.html"))


def run_test4(code_name):
    test = init(code_name)
    df = test.eval("hs04_img", "triangle")
    mdf = make_cond_mean_df(df)
    mdf["fc"] = mdf.cond.apply(lambda x: x[:5])
    mdf["img"] = mdf.cond.apply(lambda x: x[-2:])
    test4 = plot_hs04_fig11(
        mdf, max_epoch=test.cfg.total_number_of_epoch, tick_after=12
    )
    test4.save(os.path.join(test.cfg.plot_folder, "test4.html"))


def run_test5(code_name):
    tau = 1.0 / 12.0
    test = init(code_name, tau_override=tau)

    # SEM (same as HS04)
    df_intact = test.eval("train_r1000", "triangle", save_file_prefix="hi_res")
    df_os_lesion = test.eval("train_r1000", "exp_ops", save_file_prefix="hi_res")
    df_ops_lesion = test.eval("train_r1000", "ort_sem", save_file_prefix="hi_res")

    df_sem = pd.concat([df_intact, df_os_lesion, df_ops_lesion])
    mdf_sem = make_mean_df(df_sem)
    test5a = plot_hs04_fig14(mdf_sem, output="sem")
    test5a.save(os.path.join(test.cfg.plot_folder, "test5_sem.html"))

    # PHO (extra)
    df_op_lesion = test.eval("train_r1000", "exp_osp", save_file_prefix="hi_res")
    df_osp_lesion = test.eval("train_r1000", "ort_pho", save_file_prefix="hi_res")

    df_pho = pd.concat([df_intact, df_op_lesion, df_osp_lesion])
    mdf_pho = make_mean_df(df_pho)

    test5b = plot_hs04_fig14(mdf_pho, output="pho")
    test5b.save(os.path.join(test.cfg.plot_folder, "test5_pho.html"))


########## Support functions ##########


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

    metric_domain = (0, 1) if metric == "acc" else (0, mean_df.csse.max())

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
            y=alt.Y(f"mean({metric}):Q", scale=alt.Scale(domain=metric_domain)),
            color="output_name:N",
        )
        .transform_filter(interval)
    )

    return timetick_sel & main


def plot_hs04_fig10(mean_df, max_epoch, tick_after=4):
    """test case 2"""

    epoch_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=max_epoch + 1, step=10),
        fields=["epoch"],
        init={"epoch": max_epoch + 1},
        name="epoch",
    )
    sdf = mean_df.loc[(mean_df.timetick >= tick_after) & (mean_df.output_name == "pho")]

    return (
        alt.Chart(sdf)
        .mark_line()
        .encode(
            x=alt.X("freq:N", scale=alt.Scale(reverse=True)),
            y="mean(csse):Q",
            color="reg:N",
        )
        .add_selection(epoch_selection)
        .transform_filter(epoch_selection)
        .properties(width=200, height=200)
    )


def plot_conds(mean_df, tick_after=4):
    """test case 3"""

    # timetick_selection = alt.selection_single(
    #     bind=alt.binding_range(min=0, max=12, step=1),
    #     fields=["timetick"],
    #     init={"timetick": 12},
    #     name="timetick",
    # )
    sdf = mean_df.loc[(mean_df.timetick >= tick_after) & (mean_df.output_name == "pho")]

    return (
        alt.Chart(sdf)
        .mark_line()
        .encode(
            x="epoch:Q",
            y=alt.Y("mean(acc):Q", scale=alt.Scale(domain=(0, 1))),
            color="cond:N",
        )
    )

    # .add_selection(timetick_selection).transform_filter(timetick_selection)


def plot_hs04_fig11(
    mean_df,
    max_epoch,
    tick_after=4,
):
    """test case 4"""

    epoch_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=max_epoch + 1, step=10),
        fields=["epoch"],
        init={"epoch": 290},
        name="epoch",
    )
    sdf = mean_df.loc[(mean_df.timetick >= tick_after) & (mean_df.output_name == "pho")]

    return (
        alt.Chart(sdf)
        .mark_bar()
        .encode(x="img:N", y="mean(csse):Q", color="img:N", column="fc:N")
        .add_selection(epoch_selection)
        .transform_filter(epoch_selection)
        .properties(width=50, height=200)
    )


def plot_hs04_fig14(mean_df, output=None, metric="acc"):

    if output is not None:
        mean_df = mean_df.loc[(mean_df.output_name == output)]

    metric_domain = (0, 1) if metric == "acc" else (0, mean_df.csse.max())
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
            y=alt.Y(f"mean({metric}):Q", scale=alt.Scale(domain=metric_domain)),
            color="task:N",
            column="output_name:N",
        )
        .transform_filter(interval)
    )

    return timetick_sel & line

################################################################################

TEST_MAP = {1: run_test1, 2: run_test2, 3: run_test3, 4: run_test4, 5: run_test5}

if __name__ == "__main__":
    """Command line entry point, take code_name and testcase to run tests"""
    parser = argparse.ArgumentParser(description="Run HS04 test cases")
    parser.add_argument("-n", "--code_name", required=True, type=str)
    parser.add_argument("-t", "--testcase", nargs="+", type=int, help="1:EoT acc, 2:FxC, 3:NW, 4:IMG, 5:Lesion")
    args = parser.parse_args()
    [TEST_MAP[i](args.code_name) for i in args.testcase]
