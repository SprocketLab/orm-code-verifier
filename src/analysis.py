import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import wandb
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots

from src.modeling import get_model_size


def filter_and_process_data(
    df,
    metrics,
    to_plot,
    key_func=lambda x: x.split("/")[1].lower(),
):
    """
    Filter and process DataFrame for bar chart visualization.

    Args:
        df (pd.DataFrame): DataFrame containing the run data
        metrics (List[str]): List of metrics to plot
        to_plot (dict): Dictionary mapping setup names to display labels
        key_func (callable, optional): Function to process metric names

    Returns:
        pd.DataFrame: Processed DataFrame ready for plotting
    """
    # Group data by setup and model name
    flat_metrics = [m for r in metrics for m in r]
    grouped_data = df.groupby(
        ["setup", "model.name", "model.size", "generic_name", "short_name"]
    )[flat_metrics].agg(["mean", "std"])

    # Flatten multi-index columns
    grouped_data.columns = [
        f'{key_func(m)}{"(std)" if z == "std" else ""}'
        for m, z in grouped_data.columns
    ]

    # Reset index and sort by model size
    grouped_data.reset_index(inplace=True)
    grouped_data = grouped_data.sort_values(
        by="model.name", key=lambda x: [get_model_size(y) for y in x]
    )
    grouped_data = grouped_data[grouped_data["setup"].isin(to_plot.keys())]

    return grouped_data


def plot_metric(
    ax,
    metric,
    grouped_data,
    to_plot,
    colors,
    key_func=lambda x: x.split("/")[1].lower(),
    min_width_for_rotation=4,
):
    key = key_func(metric)
    bar_width = 0.8 / len(grouped_data["setup"].unique())
    n_groups = len(grouped_data["model.name"].unique())
    unique_models = grouped_data["model.name"].unique()

    # Create bars for each setup
    for j, (setup, gdf) in enumerate(grouped_data.groupby("setup")):
        if len(gdf) < n_groups:
            missing_models = set(unique_models) - set(
                gdf["model.name"].unique()
            )
            for model in missing_models:
                gdf = pd.concat(
                    [
                        gdf,
                        pd.DataFrame(
                            {
                                "model.name": [model],
                                key: [np.nan],
                                f"{key}(std)": [np.nan],
                            }
                        ),
                    ]
                )

        x = np.arange(n_groups) + j * bar_width

        rects = ax.bar(
            x, gdf[key], bar_width, label=to_plot[setup], color=colors[j]
        )

        # Choose label configuration based on label length
        max_label_length = max(
            len(f"{v:.1f}") for v in gdf[key] if not pd.isna(v)
        )
        if max_label_length > min_width_for_rotation:
            bar_label_kwargs = {
                "rotation": 90,
                "label_type": "edge",
                "padding": -30,
            }
        else:
            bar_label_kwargs = {
                "rotation": 0,
                "label_type": "center",
                "padding": -10,
            }

        # Add value labels in the middle of each bar and disable clipping
        labels = ax.bar_label(rects, fmt="%.1f", fontsize=8, **bar_label_kwargs)
        for label in labels:
            label.set_clip_on(False)

        std = gdf[f"{key}(std)"]
        if not pd.isna(std).all():
            ax.errorbar(
                x,
                gdf[key],
                yerr=std,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=5,
            )

    ax.set_xticks(
        np.arange(n_groups)
        + (bar_width * (len(grouped_data["setup"].unique()) - 1)) / 2
    )

    ax.set_xticklabels(grouped_data["short_name"].unique())

    ax.grid(True, axis="y", linestyle="--", alpha=0.2)


def create_bar_chart(
    title: str,
    subplot_titles: List[List[str]],
    metrics,
    to_plot,
    grouped_data,
    y_labels: List[List[str]],
    x_labels: List[List[str]],
    horizontal_lines=None,
    figsize=(12, 8),
    key_func=lambda x: x.split("/")[1].lower(),
    min_width_for_rotation=4,
    yscale="linear",
    hline_label: str = None,
):
    if not isinstance(metrics[0], list):
        metrics = [metrics]
    ncols = max(map(len, metrics))
    # Filter and process data
    grouped_data = filter_and_process_data(
        grouped_data, metrics, to_plot, key_func
    )

    # Calculate number of rows needed
    nrows = len(metrics)  # Ceiling division

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle case where there's only one row or column
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])

    # Set up colors
    num_groups = len(df["setup"].unique())
    colors = plt.cm.Paired(np.arange(0, num_groups))

    # Process each metric
    for ridx, row_metrics in enumerate(metrics):
        for cidx, metric in enumerate(row_metrics):

            # Plot the metric
            plot_metric(
                axes[ridx][cidx],
                metric,
                grouped_data,
                to_plot,
                colors,
                key_func,
                min_width_for_rotation,
            )

            # Set y-scale
            axes[ridx][cidx].set_yscale(yscale)
            axes[ridx][cidx].set_title(subplot_titles[ridx][cidx])
            axes[ridx][cidx].set_ylabel(y_labels[ridx][cidx])
            axes[ridx][cidx].set_xlabel(x_labels[ridx][cidx])
            if horizontal_lines and horizontal_lines[ridx][cidx] is not None:
                hline_value = horizontal_lines[ridx][cidx]
                axes[ridx][cidx].axhline(
                    y=hline_value,
                    color="black",
                    linestyle="--",
                    alpha=0.4,
                    label=hline_label,
                )
            axes[ridx, cidx].grid(True, axis="y", linestyle="--", alpha=0.2)
        # Hide any unused subplots
        for i in range(len(row_metrics), len(axes[ridx])):
            axes[ridx][i].set_visible(False)

    fig.suptitle(title)
    handles, labels = axes[0][0].get_legend_handles_labels()
    lgd = fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
    )
    fig.tight_layout(pad=1.5)

    return (fig, axes, lgd), grouped_data
