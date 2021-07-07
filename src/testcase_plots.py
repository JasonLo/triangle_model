# HS04
import pandas as pd
import numpy as np
import altair as alt

def make_mean_df(df):
    gp_vars = ['code_name', 'epoch', 'testset', 'task', 'output_name', 'timetick']
    df = df.groupby(gp_vars).mean().reset_index()
    return df

def plot_hs04_fig9(mean_df):

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
