"""
This script is used to plot the results of the simulation.
Mainly used by the benchmarks.py.
"""

import altair as alt
import pandas as pd


def plot_metric_over_epoch(
    mean_df: pd.DataFrame, output: str = None, metric: str = "acc", save: str = None
):
    """Plot selected metric over epoch with timetick selection.
    Args:
        mean_df: Dataframe with mean values of the metric over items
        output: Subset the plot by given output
            Only useful for multiple output task (triangle)
            example: "pho" or "sem"
        metric: Metric to plot
            example: "acc", "sse", "csse"
        save: Save the plot to given path
    """

    if output is not None:
        mean_df = mean_df.loc[(mean_df.output_name == output)]

    last_tick = mean_df.timetick.max()
    interval = alt.selection_interval(init={"timetick": [last_tick, last_tick]})

    metric_specific_scale = alt.Scale(domain=(0, 1)) if metric == "acc" else alt.Scale()

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

    p = timetick_sel & line

    if save:
        p.save(save)

    return p


def plot_homophony(mean_df: pd.DataFrame, save: str = None):
    """Plot Oral tasks homophony (HS04 fig.5)."""

    last_tick = mean_df.timetick.max()
    interval = alt.selection_interval(init={"timetick": [last_tick, last_tick]})

    timetick_sel = (
        alt.Chart(mean_df)
        .mark_rect()
        .encode(x="timetick:O", color=f"mean(acc):Q")
        .add_selection(interval)
    ).properties(width=400)

    scale01 = alt.Scale(domain=(0, 1))

    line = (
        alt.Chart(mean_df)
        .mark_line(point=True)
        .encode(
            x="epoch:Q",
            y=alt.Y(f"mean(acc):Q", scale=scale01),
            column=alt.Column("output_name:N", sort="descending"),
            color="cond:N",
        )
        .transform_filter(interval)
    )

    p = timetick_sel & line

    if save:
        p.save(save)

    return p
