"""
Group-Level Analysis
=====================

This example demonstrates how to run the **pyAvalanches** pipeline
on multiple subjects and aggregate results at the group level.

It follows up on the :doc:`Resting State Example </auto_examples/plot_resting_state>`
and shows how to:

1. Loop over multiple subjects
2. Assign condition labels for group comparisons
3. Aggregate results into a :class:`~pyAvalanches.subject_group_results.GroupResults` object
4. Visualize group-level ATM heatmaps
5. Generate an HTML report

Dataset
-------

We use the publicly available **EEG dataset** from OpenNeuro:

    `ds004504 v1.0.9 <https://openneuro.org/datasets/ds004504/versions/1.0.9>`_

This dataset contains resting-state EEG recordings (eyes closed)
from participants across multiple clinical groups. In this example
we select 6 subjects from 3 conditions:

- **AD** — Alzheimer's disease (subjects 001, 002, 003, 004)
- **CN** — Healthy controls (subjects 037, 038, 039, 040)
- **FTD** — Frontotemporal dementia (subjects 066, 067, 068, 069)

.. warning::

   You must download the dataset manually from OpenNeuro and update
   the ``data_dir`` variable below to point to your local copy.
"""

# %%
# Imports
# -------

import mne
from pyAvalanches.dataset import create_dataset_from_mne
from pyAvalanches.avalanches import AvalanchePipeline
from pyAvalanches.transition_matrices import ATMPipeline
from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline
from pyAvalanches.subject_group_results import Subject, GroupResults
from pyAvalanches.output import plot_group_atm_heatmap, generate_report, compute_summary_dataframes

# %%
# 1. Define Subjects and Conditions
# ------------------------------------
# We define the subject IDs and their corresponding condition labels.
# This labeling will be carried through the entire pipeline and used
# in group-level plots and reports.

subject_list = ['001', '002','003','004', '037', '038','039','040', '066', '067','068','069']
subject_labels = ['AD', 'AD','AD', 'AD', 'CN', 'CN','CN', 'CN', 'FTD', 'FTD','FTD', 'FTD']

# Update this path to your local download location
data_dir = '/Users/morgane.marzulli/Downloads'

# %%
# 2. Run the Pipeline on Each Subject
# --------------------------------------
# We loop over all subjects, applying the same processing steps:
#
# - Load the EEG data (EEGLAB ``.set`` format)
# - Detect avalanches
# - Compute the ATM
# - Extract network metrics
# - Store everything in a :class:`~pyAvalanches.subject_group_results.Subject`

subj_list = []

for index, subject_id in enumerate(subject_list):

    print(f"\n{'='*50}")
    print(f"Processing subject {subject_id} "
          f"(condition: {subject_labels[index]})")
    print(f"{'='*50}")

    # ----- Load EEG data -----
    raw = mne.io.read_raw_eeglab(
        f'{data_dir}/sub-{subject_id}_task-eyesclosed_eeg.set',
        preload=True,
        verbose=False,
    )
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} s")

    # ----- Detect avalanches -----
    avalanche_pipeline = AvalanchePipeline(
        z_scoring='by_channel',
        threshold_method='std',
        threshold=2.5,
        min_duration=2,
        min_size=1,
    )
    aval_results = avalanche_pipeline.process(raw)

    # ----- Compute ATM -----
    atm_pipeline = ATMPipeline(symm=True, method='standard')
    atm_results = atm_pipeline.process(aval_results)

    # ----- Network metrics -----
    metrics_pipeline = NetworkMetricsPipeline(
        metrics=['clustering_coeff', 'node_strength'],
        symm=True,
    )
    metrics_results = metrics_pipeline.process(atm_results)

    # ----- Store subject results -----
    sbj = Subject(
        subject_id=index,
        dataset=raw,
        subject_label=subject_labels[index],
        avalanche_results=aval_results,
        atm_results=atm_results,
        metrics_results=metrics_results,
    )
    subj_list.append(sbj)

    # Keep channel names from last subject (all should be identical)
    channels = raw.ch_names

print(f"\nDone — processed {len(subj_list)} subjects.")

# %%
# 3. Create Group Results
# -------------------------
# We aggregate all subjects into a single
# :class:`~pyAvalanches.subject_group_results.GroupResults` object.
# This is the entry point for all group-level analyses and
# visualizations.

group_results = GroupResults(
    subjects=subj_list,
    channel_names=channels,
)

print(f"Group contains {len(group_results.subjects)} subjects")
print(f"Conditions: {set(subject_labels)}")

# %%
# 4. Group-Level ATM Heatmap
# ----------------------------
# Visualize the **mean ATM per condition**. This allows you to
# compare spatiotemporal propagation patterns across clinical
# groups at a glance.
#
# pyAvalanches provides a rich set of **plotting functions** that
# return matplotlib ``Figure`` objects, so you can further customize
# them (change colormaps, add annotations, adjust layout, etc.)
# before saving.
#
# .. tip::
#
#    Check out :mod:`pyAvalanches.output` for a complete list of
#    available plotting functions, including:
#
#    - Avalanche size and duration distributions
#    - ATM heatmaps (single-subject and group-level)
#    - And more...
#
#    All of these are called internally by
#    :func:`~pyAvalanches.output.generate_report` to produce a
#    comprehensive HTML overview — but you can also use them
#    individually for fine-grained control over your figures.

plot_group_atm_heatmap(group_results)

# %%
# 5. Export Results to DataFrames
# ------------------------------
# For group-level analysis, statistical testing, or integration
# with other tools (e.g. R, Excel, seaborn), you can export all
# results into tidy `pandas <https://pandas.pydata.org>`_ DataFrames.
#
# The :func:`~pyAvalanches.output.compute_summary_dataframes` function
# takes a :class:`~pyAvalanches.subject_group_results.GroupResults`
# and returns one DataFrame per result type.


df_aval, df_pattern, df_atm, df_metrics = compute_summary_dataframes(
    group_results,
    ['avalanche', 'atm', 'metrics']
)

# %%
# Each DataFrame contains:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 25 75
#
#    * - DataFrame
#      - Contents
#    * - ``df_aval``
#      - Avalanche-level statistics (size, duration, count per epoch, etc.)
#    * - ``df_pattern``
#      - Spatiotemporal activation patterns per avalanche
#    * - ``df_atm``
#      - Flattened ATM values per epoch
#    * - ``df_metrics``
#      - Network metrics per epoch (clustering, node strength, etc.)

print("=== Avalanche Statistics ===")
print(df_aval.head())

# %%

print("=== ATM Features ===")
print(df_atm.head())

# %%

print("=== Network Metrics ===")
print(df_metrics.head())

# %%
# .. tip::
#
#    You can easily save these DataFrames for later use:
#
#    .. code-block:: python
#
#       df_aval.to_csv('./results/avalanche_stats.csv', index=False)
#       df_metrics.to_csv('./results/network_metrics.csv', index=False)

# %%
# 6. Generate Group Report
# --------------------------
# Finally, we generate a comprehensive HTML report that includes
# all subjects and conditions. The report adapts automatically
# to the :class:`~pyAvalanches.subject_group_results.GroupResults`
# input, producing per-subject panels alongside group-level
# summaries.

report = generate_report(
    results=group_results,
    out_path='./results',
    prefix='group_openneuro',
    overwrite=True,
)

print("Group report saved to ./results/group_openneuro_report.html")

# %%
# Summary
# -------
# In this example we:
#
# 1. Loaded EEG data from 6 subjects across 3 clinical conditions
# 2. Ran the full pyAvalanches pipeline on each subject
# 3. Aggregated results into a :class:`~pyAvalanches.subject_group_results.GroupResults` object
# 4. Visualized group-level ATM heatmaps
# 5. Generated an interactive HTML report
#
# Next steps
# ~~~~~~~~~~
#
# - Use :func:`~pyAvalanches.output.compute_summary_dataframes` to
#   export tidy DataFrames for statistical testing
# - Feed ATM features into a scikit-learn classifier to distinguish
#   between conditions
# - Explore different thresholds and time binnings to assess
#   robustness of the results
#
# .. seealso::
#
#    - :doc:`/auto_examples/example_resting_state` — Single-subject
#      resting-state example
#    - :doc:`/usage` — Full API walkthrough