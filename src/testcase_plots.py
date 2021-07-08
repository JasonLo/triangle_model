# HS04
import pandas as pd
import numpy as np
import altair as alt

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

def plot_hs04_fig9(mean_df):
    """test case 1"""

    timetick_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=12, step=1),
        fields=["timetick"],
        init={"timetick": 12},
        name="timetick",
    )

    return alt.Chart(mean_df).mark_line().encode(
        x='epoch:Q',
        y=alt.Y('acc:Q', scale=alt.Scale(domain=(0,1))),
        color='output_name:N'
    ).add_selection(timetick_selection).transform_filter(timetick_selection)

def plot_hs04_fig10(mean_df):
    """test case 2"""

    epoch_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=291, step=10),
        fields=["epoch"],
        init={"epoch": 290},
        name="epoch",
    )
    sdf = mean_df.loc[(mean_df.timetick >= 4) & (mean_df.output_name=='pho')]
    
    return alt.Chart(sdf).mark_line().encode(
        x=alt.X("freq:N", scale=alt.Scale(reverse=True)),
        y="mean(csse):Q",
        color="reg:N"
    ).add_selection(epoch_selection).transform_filter(epoch_selection).properties(width=200, height=200)

def plot_conds(mean_df):
    """test case 3"""

    # timetick_selection = alt.selection_single(
    #     bind=alt.binding_range(min=0, max=12, step=1),
    #     fields=["timetick"],
    #     init={"timetick": 12},
    #     name="timetick",
    # )
    sdf = mean_df.loc[(mean_df.timetick >= 4) & (mean_df.output_name=='pho')]

    return alt.Chart(sdf).mark_line().encode(
        x='epoch:Q',
        y=alt.Y('mean(acc):Q', scale=alt.Scale(domain=(0,1))),
        color='cond:N'
    )
    
    # .add_selection(timetick_selection).transform_filter(timetick_selection)

def plot_hs04_fig11(mean_df):
    """test case 4"""

    epoch_selection = alt.selection_single(
        bind=alt.binding_range(min=0, max=291, step=10),
        fields=["epoch"],
        init={"epoch": 290},
        name="epoch",
    )
    sdf = mean_df.loc[(mean_df.timetick >= 4) & (mean_df.output_name=='pho')]

    return alt.Chart(sdf).mark_bar().encode(
        x="img:N",
        y="mean(csse):Q",
        color="img:N",
        column="fc:N"
    ).add_selection(epoch_selection).transform_filter(epoch_selection).properties(width=50, height=200)
    