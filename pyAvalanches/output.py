from typing import Dict, List, Any, Optional, Union, NamedTuple
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from .subject_group_results import Subject, GroupResults

# naming convention: plot_{scope}_{measure}_{plot_type}
# scope: group (GroupResults, aggregated), per_subject (GroupResults, one panel per subject), subject (single Subject)
# measure: size, duration, atm
# plot_type: loglog, kde, heatmap

def compute_summary_dataframes(data_object: Union[Subject, GroupResults], df_type: List[str]):
    """
    Builds and returns a tuple of summary DataFrames for the specified result
    types from either a Subject or GroupResults object.

    Parameters
    ----------
    data_object : Subject or GroupResults
        The results object from which to extract summary DataFrames.
    df_type : list of str
        List of result types to extract. Allowed values are:
        - 'avalanche': returns the avalanche and patterns DataFrames.
        - 'atm': returns the ATM DataFrame.
        - 'metrics': returns the network metrics DataFrame.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing the requested DataFrames in the order they
        were specified in `df_type`. For 'avalanche', two DataFrames
        are returned (avalanche data and patterns).

    Raises
    ------
    ValueError
        If any value in `df_type` is not one of the allowed types.
    TypeError
        If `data_object` is not a Subject or GroupResults instance.
    """

    allowed_types = {'avalanche','atm','metrics'}
    # check if input metrics are okay
    for i in df_type:
        if i not in allowed_types:
            raise ValueError(
                f"df_type must be one of {allowed_types}, got '{i}'"
            )

    df_tuple = ()

    if isinstance(data_object, Subject):
        for t in df_type:
            if t == "avalanche":
                df_aval = data_object.get_avalanche_dataframe()
                df_pattern = data_object.get_patterns_dataframe()
                df_tuple = df_tuple + (df_aval, df_pattern)
                #df_tuple = df_tuple + (df_aval,)

            elif t == "atm":
                df_atm = data_object.get_atm_dataframe()
                df_tuple = df_tuple + (df_atm,)

            elif t == "metrics":
                df_metrics = data_object.get_metrics_dataframe()
                df_tuple = df_tuple + (df_metrics,)

    elif isinstance(data_object, GroupResults):
        for t in df_type:
            if t == "avalanche":
                df_aval = data_object.get_avalanche_dataframe()
                df_pattern = data_object.get_patterns_dataframe()
                df_tuple = df_tuple + (df_aval, df_pattern)

            elif t == "atm":
                df_atm = data_object.get_atm_dataframe()
                df_tuple = df_tuple + (df_atm,)

            elif t == "metrics":
                df_metrics = data_object.get_metrics_dataframe()
                df_tuple = df_tuple + (df_metrics,)
    else:
        raise TypeError("Input must be Subject or GroupResults object")

    return df_tuple


#def plot_group_log_size(group_results: GroupResults):
def plot_group_size_loglog(group_results: GroupResults):
    """
    Plots the avalanche size probability density distribution on a log-log
    scale for each group, with one subplot per group and one curve per
    condition.

    For each group, avalanche sizes are collected across all subjects and
    aggregated by condition. The probability density is computed using
    linear binning, then plotted on a log-log scale as both a scatter and
    line plot. This representation is commonly used to assess whether
    avalanche size distributions follow a power law.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Each subject must have an `avalanche_results`
        attribute with a `trial_labels` attribute and a
        `get_sizes_by_cond()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of log-log probability density plots,
        one per group, with one curve per condition.
    """

    groups = list(set(group_results.subject_labels))
    n_groups = len(groups)
    conds = list(set(group_results.subjects[0].avalanche_results.trial_labels))

    if len(groups) > 1:
        n_cols = 2
    else:
        n_cols =1
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols)  # ← shared axes
    axes = np.array(axes).flatten()
    hist_bins = 50

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    for i, group in enumerate(groups):
        ax = axes[i]
        ax.set_box_aspect(1)
        for cond in conds:

            data_to_plot = []
            for subj in group_results.subjects:
                if subj.subject_label == group:
                    # sbj_cond_data = []
                    aval_results = subj.avalanche_results
                    # conds = list(set(aval_results.trial_labels))

                    size_cond = aval_results.get_sizes_by_cond(cond=cond, merged=True)
                    data_to_plot = data_to_plot + size_cond

            Ndat = len(data_to_plot)
            xmin = np.min(data_to_plot)
            xmax = np.max(data_to_plot)

            # ── Linear Bins ────────────────────────────────────────────────────────────

            bin_counts = np.zeros(hist_bins)
            bin_centers = np.zeros(hist_bins)

            # Count occurrences in each bin
            for value in data_to_plot:
                bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
                if 1 <= bin_idx <= hist_bins:
                    bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

            for b in range(hist_bins):
                bin_centers[b] = xmin + (b + 0.5) * (xmax - xmin) / hist_bins

            # Normalized probability density
            prob_distribution = bin_counts * hist_bins / (Ndat * (xmax - xmin))

            mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

            x = bin_centers[mask]
            y = prob_distribution[mask]

            # Plot

            ax.scatter(x, y,
                       color=color_map[cond],
                       edgecolors="black",
                       s=30,
                       zorder=3,
                       label=f"Condition {cond}")

            ax.plot(x, y,
                    color=color_map[cond],
                    linewidth=1.5,
                    zorder=2)

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_xlabel("Avalanche Size", fontsize=13)
            if len(groups) > 1:
                ax.set_title(f"Group {group}", fontsize=13)
            else:
                ax.set_title(f"All Subjects", fontsize=13)

            ax.set_ylabel("Probability Density", fontsize=13)
            if len(conds) > 1:
                ax.legend(fontsize=11)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    return fig


#def plot_group_log_dur(group_results: GroupResults):
def plot_group_duration_loglog(group_results: GroupResults):
    """
    Plots the avalanche duration probability density distribution on a
    log-log scale for each group, with one subplot per group and one
    curve per condition.

    For each group, avalanche durations are collected across all subjects
    and aggregated by condition. The probability density is computed using
    linear binning, then plotted on a log-log scale as both a scatter and
    line plot. This representation is commonly used to assess whether
    avalanche duration distributions follow a power law.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Each subject must have an `avalanche_results`
        attribute with a `trial_labels` attribute and a
        `get_durations_by_cond()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of log-log probability density plots,
        one per group, with one curve per condition.
        """

    groups = list(set(group_results.subject_labels))
    n_groups = len(groups)
    conds = list(set(group_results.subjects[0].avalanche_results.trial_labels))

    if len(groups) > 1:
        n_cols = 2
    else:
        n_cols = 1
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols)
    axes = np.array(axes).flatten()
    hist_bins = 50

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    for i, group in enumerate(groups):
        ax = axes[i]
        ax.set_box_aspect(1)
        for cond in conds:

            data_to_plot = []
            for subj in group_results.subjects:
                if subj.subject_label == group:
                    # sbj_cond_data = []
                    aval_results = subj.avalanche_results
                    # conds = list(set(aval_results.trial_labels))

                    size_cond = aval_results.get_durations_by_cond(cond=cond, merged=True)
                    data_to_plot = data_to_plot + size_cond

            Ndat = len(data_to_plot)
            xmin = np.min(data_to_plot)
            xmax = np.max(data_to_plot)

            # ── Linear Bins ────────────────────────────────────────────────────────────

            bin_counts = np.zeros(hist_bins)
            bin_centers = np.zeros(hist_bins)

            # Count occurrences in each bin
            for value in data_to_plot:
                bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
                if 1 <= bin_idx <= hist_bins:
                    bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

            for b in range(hist_bins):
                bin_centers[b] = xmin + (b + 0.5) * (xmax - xmin) / hist_bins

            # Normalized probability density
            prob_distribution = bin_counts * hist_bins / (Ndat * (xmax - xmin))

            mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

            x = bin_centers[mask]
            y = prob_distribution[mask]

            # Plot

            ax.scatter(x, y,
                       color=color_map[cond],
                       edgecolors="black",
                       s=30,
                       zorder=3,
                       label=f"Condition {cond}")

            ax.plot(x, y,
                    color=color_map[cond],
                    linewidth=1.5,
                    zorder=2)

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_xlabel("Avalanche Duration", fontsize=13)
            if len(groups) > 1:
                ax.set_title(f"Group {group}", fontsize=13)
            else:
                ax.set_title(f"All Subjects", fontsize=13)

            ax.set_ylabel("Probability Density", fontsize=13)
            if len(conds) > 1:
                ax.legend(fontsize=11)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    return fig


#def plot_subj_size_distr(group_results: GroupResults):
def plot_per_subject_size_kde(group_results: GroupResults):
    """
    Plots the distribution of mean avalanche sizes for each subject, with
    one KDE curve per condition, arranged in a multi-panel grid.

    For each subject, the mean avalanche size per trial is computed for
    each condition and plotted as a KDE curve.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Each subject must have an `avalanche_results`
        attribute with a `trial_labels` attribute, a `get_sizes_by_cond()`
        method, and a `subject_id` attribute for subplot labelling.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of KDE subplots, one per subject,
        showing the mean avalanche size distribution per condition.
    """

    n_subj = group_results.n_subjects
    n_cols = 2
    n_rows = int(np.ceil(n_subj / n_cols))
    #fig, axes = plt.subplots(n_rows, n_cols)
    height_per_row = 3.5
    width_per_col = 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width_per_col * n_cols, height_per_row * n_rows),
    )
    axes = np.array(axes).flatten()

    for i, subj in enumerate(group_results.subjects):

        aval_results = subj.avalanche_results
        conds = list(set(aval_results.trial_labels))

        palette = sns.color_palette("tab10")
        color_map = {cond: palette[i] for i, cond in enumerate(conds)}

        ax = axes[i]

        for cond in conds:
            size_cond = aval_results.get_sizes_by_cond(cond=cond, merged=False)
            mean_per_trial = [np.mean(trial) for trial in size_cond]
            # data_to_plot = np.ndarray(mean_per_trial)

            sns.kdeplot(mean_per_trial, fill=True, color=color_map[cond], alpha=0.5, label=f"Condition {cond}", ax=ax)

        ax.set_xlabel("Mean Avalanche Size")
        ax.set_ylabel("Density")
        ax.set_title(f"Subject {subj.subject_id}")
        ax.legend()

    plt.tight_layout()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    return fig

#def plot_subj_duration_distr(group_results: GroupResults):
def plot_per_subject_duration_kde(group_results: GroupResults):
    """
    Plots the distribution of mean avalanche durations for each subject,
    with one KDE curve per condition, arranged in a multi-panel grid.

    For each subject, the mean avalanche duration per trial is computed
    for each condition and plotted as a KDE curve.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Each subject must have an `avalanche_results`
        attribute with a `trial_labels` attribute, a
        `get_durations_by_cond()` method, and a `subject_id` attribute
        for subplot labelling.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of KDE subplots, one per subject,
        showing the mean avalanche duration distribution per condition.
    """

    n_subj = group_results.n_subjects
    n_cols = 2
    n_rows = int(np.ceil(n_subj / n_cols))
    #fig, axes = plt.subplots(n_rows, n_cols)
    height_per_row = 3.5
    width_per_col = 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width_per_col * n_cols, height_per_row * n_rows),
    )
    axes = np.array(axes).flatten()

    for i, subj in enumerate(group_results.subjects):

        aval_results = subj.avalanche_results
        conds = list(set(aval_results.trial_labels))

        palette = sns.color_palette("tab10")
        color_map = {cond: palette[i] for i, cond in enumerate(conds)}

        ax = axes[i]

        for cond in conds:
            dur_cond = aval_results.get_durations_by_cond(cond=cond, merged=False)
            mean_per_trial = [np.mean(trial) for trial in dur_cond]
            # data_to_plot = np.ndarray(mean_per_trial)

            sns.kdeplot(mean_per_trial, fill=True, color=color_map[cond], alpha=0.5, label=f"Condition {cond}", ax=ax)

        ax.set_xlabel("Mean Avalanche Duration")
        ax.set_ylabel("Density")
        ax.set_title(f"Subject {subj.subject_id}")
        ax.legend()

    plt.tight_layout()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    return fig

#def plot_group_size_distr(group_results: GroupResults):
def plot_group_size_kde(group_results: GroupResults):
    """
    Plots the distribution of mean avalanche sizes aggregated by group,
    with one KDE curve per condition, arranged in a multi-panel grid.

    For each group, the per-subject mean avalanche size is computed for
    each condition and pooled across subjects. The resulting distribution
    is plotted as a KDE curve.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Must have a `subject_labels` attribute to
        identify groups. Each subject must have an `avalanche_results`
        attribute with a `get_sizes_by_cond()` method and a
        `subject_label` attribute for group filtering.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of KDE subplots, one per group,
        showing the mean avalanche size distribution per condition.
        """

    groups = list(set(group_results.subject_labels))
    n_groups = len(groups)
    conds = list(set(group_results.subjects[0].avalanche_results.trial_labels))

    if len(groups) > 1:
        n_cols = 2
    else:
        n_cols = 1
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols)
    axes = np.array(axes).flatten()

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    for i, group in enumerate(groups):
        ax = axes[i]
        for cond in conds:
            cond_data = []
            for subj in group_results.subjects:
                if subj.subject_label == group:
                    #sbj_cond_data = []
                    aval_results = subj.avalanche_results
                    # conds = list(set(aval_results.trial_labels))

                    size_cond = aval_results.get_sizes_by_cond(cond=cond, merged=False)
                    mean_per_trial = [np.mean(trial) for trial in size_cond]
                    all_trial_mean = np.mean(mean_per_trial)
                    # sbj_cond_data.append(all_trial_mean) # lista con le medie di tutta una cond per un soggetto

                    cond_data.append(all_trial_mean)  # tutti i soggetti di un certo gruppo

            sns.kdeplot(cond_data, fill=True, color=color_map[cond], alpha=0.5, label=f"Condition {cond}", ax=ax)

        ax.set_xlabel("Mean Avalanche Size")
        ax.set_ylabel("Density")
        if len(groups) > 1:
            ax.set_title(f"Group {group}")
        else:
            ax.set_title(f"All Subjects Average")
        ax.legend()

    plt.tight_layout()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    return fig

#def plot_group_duration_distr(group_results: GroupResults):
def plot_group_duration_kde(group_results: GroupResults):
    """
    Plots the distribution of mean avalanche durations aggregated by group,
    with one KDE curve per condition, arranged in a multi-panel grid.

    For each group, the per-subject mean avalanche duration is computed
    for each condition and pooled across subjects. The resulting
    distribution is plotted as a KDE curve.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Must have a `subject_labels` attribute to
        identify groups. Each subject must have an `avalanche_results`
        attribute with a `get_durations_by_cond()` method and a
        `subject_label` attribute for group filtering.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of KDE subplots, one per group,
        showing the mean avalanche duration distribution per condition.
    """

    groups = list(set(group_results.subject_labels))
    n_groups = len(groups)
    conds = list(set(group_results.subjects[0].avalanche_results.trial_labels))

    if len(groups) > 1:
        n_cols = 2
    else:
        n_cols = 1
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols)
    axes = np.array(axes).flatten()

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    for i, group in enumerate(groups):
        ax = axes[i]
        for cond in conds:
            cond_data = []
            for subj in group_results.subjects:
                if subj.subject_label == group:
                    # sbj_cond_data = []
                    aval_results = subj.avalanche_results
                    # conds = list(set(aval_results.trial_labels))

                    dur_cond = aval_results.get_durations_by_cond(cond=cond, merged=False)
                    mean_per_trial = [np.mean(trial) for trial in dur_cond]
                    all_trial_mean = np.mean(mean_per_trial)
                    # sbj_cond_data.append(all_trial_mean) # lista con le medie di tutta una cond per un soggetto

                    cond_data.append(all_trial_mean)  # tutti i soggetti di un certo gruppo

            sns.kdeplot(cond_data, fill=True, color=color_map[cond], alpha=0.5, label=f"Condition {cond}", ax=ax)

        ax.set_xlabel("Mean Avalanche Duration")
        ax.set_ylabel("Density")
        if len(groups) > 1:
            ax.set_title(f"Group {group}")
        else:
            ax.set_title(f"All Subjects Average")
        ax.legend()

    plt.tight_layout()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    return fig

def plot_aggregated_group_duration_kde(group_results: GroupResults):
    """
    Plots the distribution of mean avalanche durations aggregated by group,
    with all KDE curves plotted together on a single panel,
    a different color per group.

    For each group, the per-subject mean avalanche duration is computed
    for each condition and pooled across subjects. All resulting
    distributions are plotted as KDE curves on the same plot.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Must have a `subject_labels` attribute to
        identify groups. Each subject must have an `avalanche_results`
        attribute with a `get_durations_by_cond()` method and a
        `subject_label` attribute for group filtering.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a single panel showing the mean avalanche duration
        distribution for all groups and conditions with different colors per group.
    """

    groups = list(set(group_results.subject_labels))
    conds = list(set(group_results.subjects[0].avalanche_results.trial_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color palette for groups
    palette = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[2], sns.color_palette("tab10")[1]]
    color_map = {group: palette[i] for i, group in enumerate(groups)}

    # Plot KDE for each group and condition
    for group in groups:
        for cond in conds:
            cond_data = []
            for subj in group_results.subjects:
                if subj.subject_label == group:
                    aval_results = subj.avalanche_results
                    dur_cond = aval_results.get_durations_by_cond(cond=cond, merged=False)
                    mean_per_trial = [np.mean(trial) for trial in dur_cond]
                    all_trial_mean = np.mean(mean_per_trial)
                    cond_data.append(all_trial_mean)

            if cond_data:
                #sns.histplot(dur_cond[0], color=color_map[group], kde=True, binwidth=1, label=f"Group {group}")

                sns.kdeplot(
                    dur_cond[0],
                    #fill=True,
                    color=color_map[group],
                    #alpha=0.5,
                    label=f"Group {group}",
                    #cut=0,
                    ax=ax
                )

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set(xlim=(0, 80))
    ax.set_xlabel("Duration", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    #ax.set_title("Mean Avalanche Duration Distribution by Group")
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.legend()

    plt.tight_layout()

    return fig

def plot_aggregated_group_size_kde(group_results: GroupResults):
    """
    Plots the distribution of mean avalanche size aggregated by group,
    with all KDE curves plotted together on a single panel,
    a different color per group.

    For each group, the per-subject mean avalanche size is computed
    for each condition and pooled across subjects. All resulting
    distributions are plotted as KDE curves on the same plot.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        avalanche results. Must have a `subject_labels` attribute to
        identify groups. Each subject must have an `avalanche_results`
        attribute with a `get_sizes_by_cond()` method and a
        `subject_label` attribute for group filtering.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a single panel showing the mean avalanche size
        distribution for all groups and conditions with different colors per group.
    """

    groups = list(set(group_results.subject_labels))
    conds = list(set(group_results.subjects[0].avalanche_results.trial_labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color palette for groups
    #palette = sns.color_palette("tab10", len(groups))
    palette = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[2], sns.color_palette("tab10")[1]]
    color_map = {group: palette[i] for i, group in enumerate(groups)}

    # Plot KDE for each group and condition
    for group in groups:
        for cond in conds:
            cond_data = []
            for subj in group_results.subjects:
                if subj.subject_label == group:
                    aval_results = subj.avalanche_results
                    dur_cond = aval_results.get_sizes_by_cond(cond=cond, merged=False)
                    mean_per_trial = [np.mean(trial) for trial in dur_cond]
                    all_trial_mean = np.mean(mean_per_trial)
                    cond_data.append(all_trial_mean)

            if cond_data:
                #sns.histplot(dur_cond[0], color=color_map[group], kde=True, binwidth=1, label=f"Group {group}")

                sns.kdeplot(
                    dur_cond[0],
                    #fill=True,
                    color=color_map[group],
                    #alpha=0.5,
                    label=f"Group {group}",
                    ax=ax
                )

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set(xlim=(0, 500))
    ax.set_xlabel("Size", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    #ax.set_title("Mean Avalanche Size Distribution by Group")
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.legend()

    plt.tight_layout()

    return fig

#def plot_atm_heatmaps_per_subj(group_results: GroupResults):
def plot_per_subject_atm_heatmap(group_results: GroupResults):
    """
    Plots the mean ATM heatmaps for each subject and condition, arranged
    in a grid with one row per subject and one column per condition.

    The color scale (vmin, vmax) is shared within each subject across
    conditions to allow for direct visual comparison.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        ATM results. Each subject must have an `atm_results` attribute
        with a `trial_labels` attribute and a
        `get_atm_mean_by_condition()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of ATM heatmaps, with one row per
        subject and one heatmap per condition.
    """

    subjects = [sub for sub in group_results.subjects]

    # First, compute global min and max across ALL subjects and conditions
    vmin = float('inf')
    vmax = float('-inf')

    all_results = {}
    for sub in subjects:
        results_atm = sub.atm_results  # Load/get results for each subject
        all_results[sub.subject_id] = results_atm
        for cond in np.unique(results_atm.trial_labels):
            atm_mean_cond = results_atm.get_atm_mean_by_condition(cond=cond)
            vmin = min(vmin, atm_mean_cond.min())
            vmax = max(vmax, atm_mean_cond.max())

    n_conds = len(np.unique(group_results.subjects[0].atm_results.trial_labels))

    #fig = plt.figure(figsize=(4 * n_conds, 3 * len(subjects)))

    # Create one subfigure per subject
    if n_conds > 1:
        fig = plt.figure(figsize=(4 * n_conds, 3 * len(subjects)))
        subfigs = fig.subfigures(len(subjects), 1)
    else:
        fig = plt.figure(figsize=(4 * round((len(subjects)/2) + 0.5), 3 * len(subjects)))
        subfigs = fig.subfigures(len(subjects), 2)
        subfigs = subfigs.flatten()

    #subfigs = fig.subfigures(len(subjects), 1)

    for subfig, sub in zip(subfigs, subjects):
        results_atm = all_results[sub.subject_id]
        mean_ATM_list = []
        cond_labels = []

        for cond in np.unique(results_atm.trial_labels):
            atm_mean_cond = results_atm.get_atm_mean_by_condition(cond=cond)
            mean_ATM_list.append(atm_mean_cond)
            cond_labels.append(cond)

        # Set subject title on the subfigure
        subfig.suptitle(f"Subject {sub.subject_id}", fontsize=12, fontweight='bold')

        # Create inner subplots for each condition
        if n_conds > 1:
            axes = list(subfig.subplots(1, n_conds))

            # I want the barplots equal intra subject
            vmin = float('inf')
            vmax = float('-inf')
            for cond in np.unique(results_atm.trial_labels):
                atm_mean_cond = results_atm.get_atm_mean_by_condition(cond=cond)
                vmin = min(vmin, atm_mean_cond.min())
                vmax = max(vmax, atm_mean_cond.max())

        else:
            axes = [subfig.subplots(1, n_conds)]

        for ax, (matrix, cond) in zip(axes, zip(mean_ATM_list, cond_labels)):
            ax.set_box_aspect(1)
            sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
                        # xticklabels=custom_labels,
                        # yticklabels=custom_labels
                        )
            if n_conds > 1:
                ax.set_title(f"Average ATM across condition {cond}", fontsize=8)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, ha="right")
            #ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            if group_results.channel_names is not None:
                ax.set_xticks(range(len(group_results.channel_names)))
                ax.set_yticks(range(len(group_results.channel_names)))

                ax.set_xticklabels(group_results.channel_names, fontsize=12, ha="right")
                ax.set_yticklabels(group_results.channel_names, fontsize=12, ha="right")

    #plt.tight_layout()
    return fig

#def plot_atm_heatmaps_per_group(group_results: GroupResults):
def plot_group_atm_heatmap(group_results: GroupResults):
    """
    Plots the mean ATM heatmaps for each group and condition, arranged
    in a grid with one row per group and one column per condition.

    The color scale (vmin, vmax) is shared across all groups and
    conditions to allow for direct visual comparison.

    Parameters
    ----------
    group_results : GroupResults
        A GroupResults object containing all subjects and their associated
        ATM results. Must have a `subject_labels` attribute to identify
        groups and a `get_mean_atm_group()` method to retrieve the mean
        ATM for a given group and condition.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a grid of ATM heatmaps, with one row per
        group and one heatmap per condition, all sharing the same color
        scale.
    """

    groups = list(set(group_results.subject_labels))
    conds = list(set(group_results.subjects[0].atm_results.trial_labels))

    # First, compute global min and max across ALL subjects and conditions
    vmin = float('inf')
    vmax = float('-inf')

    #all_results = {}
    for group in groups:
        for cond in conds:
            atm_mean_cond = group_results.get_mean_atm_group(group=group, cond=cond)
            vmin = min(vmin, atm_mean_cond.min())
            vmax = max(vmax, atm_mean_cond.max())

    n_conds = len(conds)

    # qui sto definendo le immagini gruppo
    # fig = plt.figure(figsize=(5 * n_conds, 4 * len(groups)))

    if len(groups) > 1 and n_conds > 1:
        fig = plt.figure(figsize=(5 * n_conds, 4 * len(groups)))
        subfigs = list(fig.subfigures(len(groups), 1))

    elif len(groups) > 1 and n_conds == 1:
        fig = plt.figure(figsize=(5 * len(groups), 4))
        subfigs = list(fig.subfigures(1, len(groups)))

    elif len(groups) == 1:
        fig = plt.figure(figsize=(5 * n_conds, 4 * len(groups)))
        subfigs = [fig.subfigures(1, 1)]# ← wrap single subfigure in list

    for subfig, group in zip(subfigs, groups):
        mean_ATM_list = []
        cond_labels = []

        for cond in conds:
            atm_mean_cond = group_results.get_mean_atm_group(group=group, cond=cond)
            mean_ATM_list.append(atm_mean_cond)
            cond_labels.append(cond)

        subfig.suptitle(f"Group {group}" if len(groups) > 1 else "All Subjects",
                        fontsize=12, fontweight='bold')

        # qui sono all'interno dell'immagine gruppo
        if n_conds > 1:
            axes = list(subfig.subplots(1, n_conds))
        else:
            axes = [subfig.subplots(1, 1)]  # ← wrap single ax in list

        for ax, (matrix, cond) in zip(axes, zip(mean_ATM_list, cond_labels)):
            sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax)
            if n_conds > 1:
                ax.set_title(f"Average ATM across condition {cond}")
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, ha="right")
            #ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
            if group_results.channel_names is not None:
                ax.set_xticklabels(group_results.channel_names, rotation=90, fontsize=12, ha="right")
                ax.set_yticklabels(group_results.channel_names, fontsize=12, ha="right")

    return fig

#def plot_single_subj_log_size(subject: Subject):
def plot_subject_size_loglog(subject: Subject):
    """
    Plots the avalanche size probability density distribution on a log-log
    scale for a single subject, with one curve per condition.

    The probability density is computed using linear binning and plotted
    as both a scatter and line plot on a log-log scale.

    Parameters
    ----------
    subject : Subject
        A Subject object containing avalanche results in its
        `avalanche_results` attribute, with a `trial_labels` attribute
        and a `get_sizes_by_cond()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a single log-log probability density plot
        of avalanche sizes, with one curve per condition.
    """

    conds = list(set(subject.avalanche_results.trial_labels))


    fig, ax = plt.subplots(1, 1)
    hist_bins = 50

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    #for i, group in enumerate(groups):
    #    ax = axes[i]
    ax.set_box_aspect(1)
    for cond in conds:
        # sbj_cond_data = []
        aval_results = subject.avalanche_results
        # conds = list(set(aval_results.trial_labels))

        data_to_plot = aval_results.get_sizes_by_cond(cond=cond, merged=True)
        #data_to_plot = size_cond

        Ndat = len(data_to_plot)
        xmin = np.min(data_to_plot)
        xmax = np.max(data_to_plot)

        # ── Linear Bins ────────────────────────────────────────────────────────────

        bin_counts = np.zeros(hist_bins)
        bin_centers = np.zeros(hist_bins)

        # Count occurrences in each bin
        for value in data_to_plot:
            bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
            if 1 <= bin_idx <= hist_bins:
                bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

        for b in range(hist_bins):
            bin_centers[b] = xmin + (b + 0.5) * (xmax - xmin) / hist_bins

        # Normalized probability density
        prob_distribution = bin_counts * hist_bins / (Ndat * (xmax - xmin))

        mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

        x = bin_centers[mask]
        y = prob_distribution[mask]

        # Plot

        ax.scatter(x, y,
                   color=color_map[cond],
                   edgecolors="black",
                   s=30,
                   zorder=3,
                   label=f"Condition {cond}")

        ax.plot(x, y,
                color=color_map[cond],
                linewidth=1.5,
                zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Avalanche Size", fontsize=13)
        #ax.set_title(f"Subject ", fontsize=13)

        ax.set_ylabel("Probability Density", fontsize=13)
        if len(conds) > 1:
            ax.legend(fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig

#def plot_single_subj_log_duration(subject: Subject):
def plot_subject_duration_loglog(subject: Subject):
    """
    Plots the avalanche duration probability density distribution on a
    log-log scale for a single subject, with one curve per condition.

    The probability density is computed using linear binning and plotted
    as both a scatter and line plot on a log-log scale.

    Parameters
    ----------
    subject : Subject
        A Subject object containing avalanche results in its
        `avalanche_results` attribute, with a `trial_labels` attribute
        and a `get_durations_by_cond()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a single log-log probability density plot
        of avalanche durations, with one curve per condition.
    """


    conds = list(set(subject.avalanche_results.trial_labels))


    fig, ax = plt.subplots(1, 1)
    hist_bins = 50

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    #for i, group in enumerate(groups):
    #    ax = axes[i]
    ax.set_box_aspect(1)
    for cond in conds:
        # sbj_cond_data = []
        aval_results = subject.avalanche_results
        # conds = list(set(aval_results.trial_labels))

        data_to_plot = aval_results.get_durations_by_cond(cond=cond, merged=True)
        #data_to_plot = size_cond

        Ndat = len(data_to_plot)
        xmin = np.min(data_to_plot)
        xmax = np.max(data_to_plot)

        # ── Linear Bins ────────────────────────────────────────────────────────────

        bin_counts = np.zeros(hist_bins)
        bin_centers = np.zeros(hist_bins)

        # Count occurrences in each bin
        for value in data_to_plot:
            bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
            if 1 <= bin_idx <= hist_bins:
                bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

        for b in range(hist_bins):
            bin_centers[b] = xmin + (b + 0.5) * (xmax - xmin) / hist_bins

        # Normalized probability density
        prob_distribution = bin_counts * hist_bins / (Ndat * (xmax - xmin))

        mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

        x = bin_centers[mask]
        y = prob_distribution[mask]

        # Plot

        ax.scatter(x, y,
                   color=color_map[cond],
                   edgecolors="black",
                   s=30,
                   zorder=3,
                   label=f"Condition {cond}")

        ax.plot(x, y,
                color=color_map[cond],
                linewidth=1.5,
                zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Avalanche Duration", fontsize=13)
        #ax.set_title(f"Subject ", fontsize=13)

        ax.set_ylabel("Probability Density", fontsize=13)
        if len(conds) > 1:
            ax.legend(fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig

#def plot_single_subj_atm_heatmap(subject: Subject):
def plot_subject_atm_heatmap(subject: Subject):
    """
    Plots the mean ATM heatmaps for a single subject, with one heatmap
    per condition arranged in a single row.

    The color scale (vmin, vmax) is shared across all conditions to
    allow for direct visual comparison.

    Parameters
    ----------
    subject : Subject
        A Subject object containing ATM results in its `atm_results`
        attribute, with a `trial_labels` attribute and a
        `get_atm_mean_by_condition()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a row of ATM heatmaps, one per condition,
        all sharing the same color scale.
    """

    mean_ATM_list = []
    cond_list = list(set(subject.atm_results.trial_labels))

    for cond in cond_list:
        atm_mean_cond = subject.atm_results.get_atm_mean_by_condition(cond=cond)
        mean_ATM_list.append(atm_mean_cond)

    #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin = min(matrix.min() for matrix in mean_ATM_list)
    vmax = max(matrix.max() for matrix in mean_ATM_list)

    #fig, axes = plt.subplots(1, len(mean_ATM_list), figsize=(10, 4), squeeze=False)
    fig, axes = plt.subplots(1, len(mean_ATM_list), figsize=(4*len(mean_ATM_list), 3), squeeze=False)
    axes = axes[0]

    for ax, (matrix, cond) in zip(axes, zip(mean_ATM_list, cond_list)):
        ax.set_box_aspect(1)
        sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax)

        if subject.dataset is not None:
            if subject.dataset['channel_names'] is not None:
                num_channels = len(subject.dataset['channel_names'])

                # Dynamically determine label spacing
                if num_channels > 30:
                    # Show ~10 labels max
                    step = num_channels // 10
                elif num_channels > 15:
                    # Show ~15 labels max
                    step = max(1, num_channels // 15)
                else:
                    # Show all labels
                    step = 1

                # Set ticks and labels
                tick_positions = range(0, num_channels, step)
                tick_labels = [subject.dataset['channel_names'][i] for i in tick_positions]

                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=12)
                ax.set_yticklabels(tick_labels, fontsize=12)

        if len(cond_list) > 1:
                ax.set_title(f"Condition {cond}")


    #plt.close(fig)
    plt.tight_layout()
    return fig

#def plot_single_subj_size_distr(subject: Subject):
def plot_subject_size_kde(subject: Subject):
    """
    Plots the distribution of mean avalanche sizes for a single subject,
    with one KDE curve per condition, on a single plot.

    The mean avalanche size per trial is computed for each condition and
    plotted as a KDE curve.

    Parameters
    ----------
    subject : Subject
        A Subject object containing avalanche results in its
        `avalanche_results` attribute, with a `trial_labels` attribute
        and a `get_sizes_by_cond()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a single KDE plot showing the mean avalanche
        size distribution per condition.
    """

    aval_results = subject.avalanche_results
    conds = list(set(aval_results.trial_labels))

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for cond in conds:
        size_cond = aval_results.get_sizes_by_cond(cond=cond, merged=False)
        mean_per_trial = [np.mean(trial) for trial in size_cond]
        # data_to_plot = np.ndarray(mean_per_trial)

        sns.kdeplot(mean_per_trial, fill=True, color=color_map[cond], alpha=0.5, label=f"Condition {cond}", ax=ax)

    ax.set_xlabel("Mean Avalanche Size")
    ax.set_ylabel("Density")
    ax.legend()

    #plt.close(fig)

    return fig


#def plot_single_subj_duration_distr(subject: Subject):
def plot_subject_duration_kde(subject: Subject):
    """
    Plots the distribution of mean avalanche durations for a single
    subject, with one KDE curve per condition, on a single plot.

    The mean avalanche duration per trial is computed for each condition
    and plotted as a KDE curve.

    Parameters
    ----------
    subject : Subject
        A Subject object containing avalanche results in its
        `avalanche_results` attribute, with a `trial_labels` attribute
        and a `get_durations_by_cond()` method.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure containing a single KDE plot showing the mean avalanche
        duration distribution per condition.
    """

    aval_results = subject.avalanche_results
    conds = list(set(aval_results.trial_labels))

    palette = sns.color_palette("tab10")
    color_map = {cond: palette[i] for i, cond in enumerate(conds)}

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for cond in conds:
        dur_cond = aval_results.get_durations_by_cond(cond=cond, merged=False)
        mean_per_trial = [np.mean(trial) for trial in dur_cond]
        # data_to_plot = np.ndarray(mean_per_trial)

        sns.kdeplot(mean_per_trial, fill=True, color=color_map[cond], alpha=0.5, label=f"Condition {cond}", ax=ax)

    ax.set_xlabel("Mean Avalanche Duration")
    ax.set_ylabel("Density")
    ax.legend()

    #plt.close(fig)

    return fig

#def make_df_html(df_input):
def _render_dataframe_html(df_input):
    """
    Converts a pandas DataFrame to a styled HTML string, dropping any
    columns containing list or array values.

    Applies Bootstrap-compatible table classes and custom inline CSS
    for use in MNE HTML reports or other HTML contexts.

    Parameters
    ----------
    df_input : pd.DataFrame
        The DataFrame to render. Columns containing list or np.ndarray
        values are automatically excluded from the output.

    Returns
    -------
    str
        A styled HTML string representing the filtered DataFrame,
        wrapped in a styled div container.
    """
    # Drop columns where ANY cell contains a list
    cols_to_keep = [col for col in df_input.columns
                    if not df_input[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any()]

    df_filtered = df_input[cols_to_keep]

    df_html = df_filtered.to_html(
        classes='table table-striped table-hover',  # Bootstrap classes (MNE uses Bootstrap)
        border=0,
        index=False,       # Hide row index
        float_format='{:.4f}'.format  # Format floats
    )

    # Wrap in a styled div
    styled_html = f"""
    <div style="margin: 20px; overflow-x: auto;">
        <style>
            .mne-df table {{ border-collapse: collapse; width: 80%; font-size: 12px; }}
            .mne-df th {{ background-color: #e0e0e0; color: black !important; 
                          padding: 5px 8px; border: 1px solid #ddd; text-align: left !important; }}
            .mne-df td {{ padding: 5px 8px; border: 1px solid #ddd; color: black !important; text-align: left; }}
            .mne-df tbody tr:nth-child(odd)  {{ background-color: #ffffff !important; }}
            .mne-df tbody tr:nth-child(even) {{ background-color: #d3d3d3 !important; }}
        </style>
        <div class="mne-df">
            {df_html}
        </div>
    </div>
    """

    return styled_html

def generate_report(
    results: Union[GroupResults, Subject],
    out_path: str,
    prefix: Union[str, None] = None,
    overwrite: bool = False,
) -> mne.Report:
    """
    Generates and saves an MNE HTML report for neuronal avalanche analysis.

    Behaviour adapts automatically depending on whether group-level or
    subject-level results are provided:
    - GroupResults → group-level report including per-subject panels.
    - Subject → subject-level report only.

    The report includes summary tables, avalanche size and duration
    distributions (log-log and KDE), and ATM heatmaps, depending on
    which results are available in the input object.

    Parameters
    ----------
    results : GroupResults or Subject
        Results object from the avalanche pipeline. The type is detected
        automatically to switch between group-level and subject-level
        reporting.
    out_path : str
        Path to the directory where the report will be saved.
    prefix : str or None, optional
        Prefix string used to name the output file. The report will be
        saved as "{prefix}_report.html". If None, the prefix is set
        automatically based on the input type:
        - GroupResults → 'group'
        - Subject → 'subject'
        Defaults to None.
    overwrite : bool, optional
        If True, overwrites any existing report with the same filename.
        Defaults to False.

    Returns
    -------
    mne.Report
        The populated MNE Report object.

    Raises
    ------
    TypeError
        If `results` is neither a GroupResults nor a Subject instance.
    """

    #  detect input type
    if isinstance(results, GroupResults):
        level = "group"
    elif isinstance(results, Subject):
        level = "subject"
    else:
        raise TypeError(
            f"Expected a GroupResults or SubjectResults object, "
            f"got {type(results).__name__!r} instead."
        )

    #  set filename based on level if not provided
    if prefix is None:
        prefix = f"{level}"

    #  initialise report
    report_title = (
        "Neural Avalanche Analysis Report – Group Level"
        if level == "group"
        else "Neural Avalanche Analysis Report – Subject Level"
    )
    report_out = mne.Report(title=report_title)

    #  Branch: group-level report
    if level == "group":

        # --- Summary tables ----------------------------------------------- #
        if results.all_subjects_have("dataset"):
            df_dataset = results.get_dataset_dataframe()
            html_dataset = _render_dataframe_html(df_dataset)

            channel_names = results.subjects[0].dataset['channel_names']
            channel_text = f"<p><strong>Channel Names:</strong> {', '.join(channel_names)}</p>"
            html_dataset = html_dataset + channel_text

            report_out.add_html(
                html=html_dataset,
                title="Summary Table – Dataset Info",
                tags=("tables", "dataset"),
            )

        if results.all_subjects_have("avalanche_results"):
            df_aval = results.get_avalanche_dataframe()
            html_aval = _render_dataframe_html(df_aval)

            report_out.add_html(
                html=html_aval,
                title="Summary Table – Avalanche Data",
                tags=("tables", "avalanches"),
            )

        if results.all_subjects_have("metrics_results"): # check if the metrics were computed or not
            df_metrics = results.get_metrics_dataframe() # placeholder
            html_metrics = _render_dataframe_html(df_metrics)

            report_out.add_html(
                html=html_metrics,
                title="Summary Table – ATM Network Metrics",
                tags=("tables", "metrics"),
            )

        # --- Log-scale distributions --------------------------------------- #
        # these plots are made regardless of the number of groups and conditions
        if results.all_subjects_have("avalanche_results"):
            fig_log_size = plot_group_size_loglog(results)
            fig_log_dur = plot_group_duration_loglog(results)

            report_out.add_figure(
                fig=fig_log_size,
                title="Avalanche Size Distribution – Group Level (Log Scale)",
                caption=(
                    "Log-log plot of the avalanche size distribution "
                    "averaged across all subjects in the group. "
                ),
                tags=("distributions", "size", "log-scale"),
                image_format="PNG",
            )

            report_out.add_figure(
                fig=fig_log_dur,
                title="Avalanche Duration Distribution – Group Level (Log Scale)",
                caption=(
                    "Log-log plot of the avalanche duration distribution "
                    "averaged across all subjects in the group. "
                ),
                tags=("distributions", "duration", "log-scale"),
                image_format="PNG",
            )

        # --- Size distributions (KDE) ----------------------------- #
            if len(set(results.subjects[0].avalanche_results.trial_labels)) > 1: # if I have multiple trial conditions

                fig_group_size = plot_group_size_kde(results)
                fig_subj_size = plot_per_subject_size_kde(results)

                report_out.add_figure(
                    fig=fig_group_size,
                    title="Avalanche Size Distribution – Group Average (KDE)",
                    caption=(
                        "Avalanche size distribution averaged across all subjects (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "size", "group"),
                    image_format="PNG",
                )
                report_out.add_figure(
                    fig=fig_subj_size,
                    title="Avalanche Size Distribution – Per Subject (KDE)",
                    caption=(
                        "Avalanche size distribution shown separately for each subject (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "size", "subject"),
                    image_format="PNG",
                )

                # --- Duration distributions (KDE) ------------------------- #
                fig_group_dur = plot_group_duration_kde(results)
                fig_subj_dur = plot_per_subject_duration_kde(results)

                report_out.add_figure(
                    fig=fig_group_dur,
                    title="Avalanche Duration Distribution – Group Average (KDE)",
                    caption=(
                        "Avalanche duration distribution averaged across all subjects (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "duration", "group"),
                    image_format="PNG",
                )
                report_out.add_figure(
                    fig=fig_subj_dur,
                    title="Avalanche Duration Distribution – Per Subject (KDE)",
                    caption=(
                        "Avalanche duration distribution shown separately for each subject (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "duration", "subject"),
                    image_format="PNG",
                )

            elif results.subjects[0].avalanche_results.trial_count == 1: # resting state data
                figure_kde_grouped_size = plot_aggregated_group_size_kde(results)
                report_out.add_figure(
                    fig = figure_kde_grouped_size,
                    title = "Avalanche Size Distribution – Group Average (KDE)",
                    caption = (
                        "Avalanche size distribution averaged across all subjects for each group (Kernel Density Estimation plot). "
                    ),
                    tags = ("distributions", "size", "group"),
                    image_format = "PNG",
                )

                figure_kde_grouped_duration = plot_aggregated_group_duration_kde(results)
                report_out.add_figure(
                    fig=figure_kde_grouped_duration,
                    title="Avalanche Duration Distribution – Group Average (KDE)",
                    caption=(
                        "Avalanche duration distribution averaged across all subjects for each group (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "duration", "group"),
                    image_format="PNG",
                )



        # --- ATM heatmaps -------------------------------------------------- #
        if results.all_subjects_have("atm_results"):
            fig_atm_group = plot_group_atm_heatmap(results)
            fig_atm_subj = plot_per_subject_atm_heatmap(results)

            report_out.add_figure(
                fig=fig_atm_group,
                title="Avalanche Transition Matrix (ATM) – Group Level",
                caption=(
                    "Heatmap of the Avalanche Transition Matrix averaged within each group. "
                    "Each cell (i, j) encodes the probability that channel j "
                    "activates immediately after channel i within the same avalanche."
                ),
                tags=("ATM", "group"),
                image_format="PNG",
            )
            report_out.add_figure(
                fig=fig_atm_subj,
                title="Avalanche Transition Matrix (ATM) – Per Subject",
                caption=(
                    "Heatmap of the Avalanche Transition Matrix for each individual subject. "
                ),
                tags=("ATM", "subject"),
                image_format="PNG",
            )


    #  Branch: subject-level report
    elif level == "subject":

        if results.dataset is not None:
            df_dataset = results.get_dataset_dataframe()
            html_dataset = _render_dataframe_html(df_dataset)

            channel_names = results.dataset['channel_names']
            channel_text = f"<p><strong>Channel Names:</strong> {', '.join(channel_names)}</p>"
            html_dataset = html_dataset + channel_text

            report_out.add_html(
                html=html_dataset,
                title="Summary Table – Dataset Info",
                tags=("tables", "dataset"),
            )

        if results.avalanche_results is not None:
            df_aval = results.get_avalanche_dataframe()  # placeholder

            html_aval = _render_dataframe_html(df_aval)
            report_out.add_html(
                html=html_aval,
                title="Summary Table – Avalanche Data",
                tags=("tables", "avalanches"),
            )

        if results.metrics_results is not None:  # check if the metrics were computed or not
            df_metrics = results.get_metrics_dataframe()  # placeholder
            html_metrics = _render_dataframe_html(df_metrics)

            report_out.add_html(
                html=html_metrics,
                title="Summary Table – ATM Network Metrics",
                tags=("tables", "metrics"),
            )

        if results.avalanche_results is not None:

            n_conds = len(set(results.avalanche_results.trial_labels))

            fig_log_size = plot_subject_size_loglog(results)
            fig_log_duration = plot_subject_duration_loglog(results)

            report_out.add_figure(
                fig=fig_log_size,
                title="Avalanche Size Distribution – Subject Level (Log Scale)",
                caption=(
                    "Log-log plot of the avalanche size distribution "
                    "for the individual subject. "
                ),
                tags=("distributions", "size", "log-scale"),
                image_format="PNG",
            )

            report_out.add_figure(
                fig=fig_log_duration,
                title="Avalanche Duration Distribution – Subject Level (Log Scale)",
                caption=(
                    "Log-log plot of the avalanche duration distribution "
                    "for the individual subject. "
                ),
                tags=("distributions", "duration", "log-scale"),
                image_format="PNG",
            )

            if n_conds > 1:

                fig_size = plot_subject_size_kde(results)
                fig_dur = plot_subject_duration_kde(results)

                report_out.add_figure(
                    fig=fig_size,
                    title="Avalanche Size Distribution – Subject Level",
                    caption=(
                        "Avalanche size distribution for the individual subject (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "size", "subject"),
                    image_format="PNG",
                )

                report_out.add_figure(
                    fig=fig_dur,
                    title="Avalanche Duration Distribution – Subject Level",
                    caption=(
                        "Avalanche duration distribution for the individual subject (Kernel Density Estimation plot). "
                    ),
                    tags=("distributions", "duration", "subject"),
                    image_format="PNG",
                )

        if results.atm_results is not None:

            fig_atm = plot_subject_atm_heatmap(results)

            report_out.add_figure(
                fig=fig_atm,
                title="Avalanche Transition Matrix (ATM) – Subject Level",
                caption=(
                    "Heatmap of the Avalanche Transition Matrix for the individual subject. "
                    "Each cell (i, j) encodes the probability that channel j "
                    "activates immediately after channel i within the same avalanche. "
                ),
                tags=("ATM", "subject"),
                image_format="PNG",
            )

    #  Save
    plt.close('all')
    report_out.save(fname=os.path.join(out_path, f"{prefix}_report.html"), overwrite=overwrite)
    print(f"[generate_avalanche_report] Report saved in {out_path}")

    return report_out