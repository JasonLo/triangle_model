"""
This script is used to plot the results of the simulation.
Mainly used by the benchmarks.py.
"""

import altair as alt
import pandas as pd
from PIL import Image
from typing import List


def plot_metric_over_epoch(
    mean_df: pd.DataFrame, output: str = None, metric: str = "acc", save: str = None
) -> alt.Chart:
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


def plot_homophony(mean_df: pd.DataFrame, save: str = None) -> alt.Chart:
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


def plot_triangle(mean_df: pd.DataFrame, metric="acc", save: str = None) -> alt.Chart:
    """Plotting the metric over epoch by output (HS04, fig 9)."""

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
            color="output_name:N",
            tooltip=[f"mean({metric})"],
        )
        .interactive()
        .transform_filter(interval)
    )

    p = timetick_sel & line

    if save:
        p.save(save)
    return p


def stitch_fig(images: List[str], rows: int, columns: int) -> Image:
    """Stitch images in a grid."""
    assert len(images) <= (rows * columns)

    images = [Image.open(x) for x in images]

    # All images dimensions
    widths, heights = zip(*(im.size for im in images))

    # Max dims
    max_width = max(widths)
    max_height = max(heights)

    # Stitching
    stitched_image = Image.new("RGB", (max_width * columns, max_height * rows))

    x_offset = 0
    y_offset = 0

    for i, im in enumerate(images):
        stitched_image.paste(im, (x_offset, y_offset))
        if (i + 1) % columns == 0:
            # New row every {columns} images
            y_offset += max_height
            x_offset = 0
        else:
            # New column
            x_offset += max_width

    return stitched_image
