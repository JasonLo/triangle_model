import argparse, os
import pandas as pd
import altair as alt
import meta, modeling, evaluate

def main(code_name, testcase=None):
    """Command line entry point"""
    test_obj = init(code_name)
    
    if testcase is None:
        [TEST_MAP[i+1](test_obj) for i in range(5)]
    else:
        TEST_MAP[testcase](test_obj)
    
def init(code_name):
    cfg = meta.ModelConfig.from_json(
        os.path.join("models", code_name, "model_config.json")
    )
    model = modeling.MyModel(cfg)
    test = evaluate.TestSet(cfg, model)
    return test

def run_test1(test):
    df = test.eval("train_r1000", "triangle")
    mdf =  make_mean_df(df)
    fig9 =  plot_hs04_fig9(mdf, test.cfg.n_timesteps)
    fig9.save(os.path.join(test.cfg.path["plot_folder"], "test1.html"))
    
def run_test2(test):
    df = test.eval("taraban", "triangle")
    mdf =  make_cond_mean_df(df)

    # TODO: Refractorized testset specific post-processing
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

    fig10 =  plot_hs04_fig10(mdf, max_epoch=test.cfg.total_number_of_epoch, tick_after=12)
    fig10.save(os.path.join(test.cfg.path["plot_folder"], "test2.html"))
    

def run_test3(test):
    df = test.eval("glushko", "triangle")
    mdf =  make_cond_mean_df(df)
    test3 =  plot_conds(mdf, tick_after=12)
    test3.save(os.path.join(test.cfg.path["plot_folder"], "test3.html"))
    

def run_test4(test):
    df = test.eval("hs04_img", "triangle")
    mdf =  make_cond_mean_df(df)
    mdf["fc"] = mdf.cond.apply(lambda x: x[:5])
    mdf["img"] = mdf.cond.apply(lambda x: x[-2:])
    test4 =  plot_hs04_fig11(mdf, max_epoch=test.cfg.total_number_of_epoch, tick_after=12)
    test4.save(os.path.join(test.cfg.path["plot_folder"], "test4.html"))
    

def run_test5(test):
    # SEM (same as HS04)
    df_intact = test.eval("train_r1000", "triangle")
    df_os_lesion = test.eval("train_r1000", "exp_ops")
    df_ops_lesion = test.eval("train_r1000", "ort_sem")

    df = pd.concat([df_intact, df_os_lesion, df_ops_lesion])
    mdf =  make_mean_df(df)

    test5a =  plot_hs04_fig14(mdf, output="sem")
    test5a.save(os.path.join(test.cfg.path["plot_folder"], "test5_sem.html"))
    
    # PHO (extra)
    df_op_lesion = test.eval("train_r1000", "exp_osp")
    df_osp_lesion = test.eval("train_r1000", "ort_pho")
    
    df = pd.concat([df_intact, df_op_lesion, df_osp_lesion])
    mdf =  make_mean_df(df)

    test5b =  plot_hs04_fig14(mdf, output='pho')
    test5b.save(os.path.join(test.cfg.path["plot_folder"], "test5_pho.html"))
    
TEST_MAP = {
    1: run_test1,
    2: run_test2,
    3: run_test3,
    4: run_test4,
    5: run_test5
}    
   
########## Support functions ##########

def print_unique(df):
    for x in ("testset", "task", "output_name", "timetick", "cond"):
        print(f"{x}: unique entry: {df[x].unique()}")

def make_mean_df(df):
    """Aggregate on items axis to one value"""
    gp_vars = ['code_name', 'epoch', 'testset', 'task', 'output_name', 'timetick']
    df = df.groupby(gp_vars).mean().reset_index()
    return df

def make_cond_mean_df(df):
    """Aggregate on items axis with condition"""
    df['csse'] = df.sse.loc[df.acc == 1]
    gp_vars = ['code_name', 'epoch', 'testset', 'task', 'output_name', 'timetick', 'cond']
    df = df.groupby(gp_vars).mean().reset_index()
    return df

def plot_hs04_fig9(mean_df, steps=12):
    """test case 1"""

    timetick_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=steps, step=1),
        fields=["timetick"],
        init={"timetick": 12},
        name="timetick",
    )

    return alt.Chart(mean_df).mark_line().encode(
        x='epoch:Q',
        y=alt.Y('acc:Q', scale=alt.Scale(domain=(0,1))),
        color='output_name:N'
    ).add_selection(timetick_selection).transform_filter(timetick_selection)

def plot_hs04_fig10(mean_df, max_epoch, tick_after=4):
    """test case 2"""

    epoch_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=max_epoch+1, step=10),
        fields=["epoch"],
        init={"epoch": 290},
        name="epoch",
    )
    sdf = mean_df.loc[(mean_df.timetick >= tick_after) & (mean_df.output_name=='pho')]
    
    return alt.Chart(sdf).mark_line().encode(
        x=alt.X("freq:N", scale=alt.Scale(reverse=True)),
        y="mean(csse):Q",
        color="reg:N"
    ).add_selection(epoch_selection).transform_filter(epoch_selection).properties(width=200, height=200)

def plot_conds(mean_df, tick_after=4):
    """test case 3"""

    # timetick_selection = alt.selection_single(
    #     bind=alt.binding_range(min=0, max=12, step=1),
    #     fields=["timetick"],
    #     init={"timetick": 12},
    #     name="timetick",
    # )
    sdf = mean_df.loc[(mean_df.timetick >= tick_after) & (mean_df.output_name=='pho')]

    return alt.Chart(sdf).mark_line().encode(
        x='epoch:Q',
        y=alt.Y('mean(acc):Q', scale=alt.Scale(domain=(0,1))),
        color='cond:N'
    )
    
    # .add_selection(timetick_selection).transform_filter(timetick_selection)

def plot_hs04_fig11(mean_df, max_epoch, tick_after=4, ):
    """test case 4"""

    epoch_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=max_epoch+1, step=10),
        fields=["epoch"],
        init={"epoch": 290},
        name="epoch",
    )
    sdf = mean_df.loc[(mean_df.timetick >= tick_after) & (mean_df.output_name=='pho')]

    return alt.Chart(sdf).mark_bar().encode(
        x="img:N",
        y="mean(csse):Q",
        color="img:N",
        column="fc:N"
    ).add_selection(epoch_selection).transform_filter(epoch_selection).properties(width=50, height=200)
    

def plot_hs04_fig14(mean_df, output):

    mean_df = mean_df.loc[(mean_df.output_name == output)]
    print_unique(mean_df)

    interval = alt.selection_interval()

    timetick_sel = (
        alt.Chart(mean_df).mark_rect().encode(
            x="timetick:O",
            color="mean(acc):Q"
        ).add_selection(interval)
    ).properties(width=400)

    line = (
        alt.Chart(mean_df)
        .mark_line(point=True)
        .encode(
            x="epoch:Q",
            y=alt.Y("mean(acc):Q", scale=alt.Scale(domain=(0, 1))),
            color="task:N"
        )
        .transform_filter(interval)
    )

    return timetick_sel & line

################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run HS04 test cases')
    parser.add_argument("code_name")
    parser.add_argument("testcase", help="hs04 testcase 1:EoT acc, 2:FxC, 3:NW, 4:IMG, 5:Lesion", type=int)
    args = parser.parse_args()
    main(**vars(args))

