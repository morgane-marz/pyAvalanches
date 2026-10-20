"""
Resting State EEG Example
===========================

This example demonstrates the full **pyAvalanches** pipeline applied
to a single-subject EEG resting-state recording, from data loading
to avalanche transition matrix visualization.

Dataset
-------

We use the publicly available **EEG dataset** from OpenNeuro:

    `ds004504 v1.0.9 <https://openneuro.org/datasets/ds004504/versions/1.0.9>`_

This dataset contains resting-state EEG recordings (eyes closed)
from participants across multiple clinical groups. In this example
we analyze a single subject.

.. warning::

   You must download the dataset manually from OpenNeuro and update
   the ``data_dir`` variable below to point to your local copy.

.. seealso::

   :doc:`/auto_examples/example_group_analysis` — Extend this workflow
   to multiple subjects and group-level comparisons.
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import openneuro
from pathlib import Path

from pyAvalanches.dataset import plot_amplitude_distribution
from pyAvalanches.avalanches import AvalanchePipeline, plot_avalanche_distribution
from pyAvalanches.transition_matrices import ATMPipeline
from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline
from pyAvalanches.subject_group_results import Subject
from pyAvalanches.output import generate_report, compute_summary_dataframes

# %%
# 1. Load EEG Data
# ------------------
# We load a single resting-state EEG recording (eyes closed)
# in EEGLAB ``.set`` format.


DATA_DIR = Path(mne.get_config("MNE_DATA") or (Path.home() / "mne_data")) / "ds004504"
TASK = "eyesclosed"
SUBDIR = "derivatives"

openneuro.download(
    dataset="ds004504",
    target_dir=DATA_DIR,
    include=["derivatives/sub-001/eeg/*"],
)

raw = mne.io.read_raw_eeglab(
    DATA_DIR / SUBDIR / "sub-001" / "eeg" / "sub-001_task-eyesclosed_eeg.set",
    preload=True,
    verbose=False,
)

print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} s")


# %%
# 2. Z-Score and Binarize
# -------------------------
# The data is z-scored across the entire recording, then binarized
# using a fixed threshold. Each time-channel point above the
# threshold is marked as active (1), otherwise inactive (0). Visualizing the amplitudes distributions
# across channels can help selecting the right threshold for binarization.


plot_amplitude_distribution(raw)


# %%
# 3. Avalanche Distributions Across Time Binnings
# --------------------------------------------------
# A key signature of criticality is that avalanche size and duration
# distributions should follow a power law.
#
# Here we compute and plot the distributions for multiple time binnings
# (1, 2, 3, 5, and 10 samples per bin) to verify this scaling behavior.
#
# .. note::
#
#    The ``linear=1`` parameter plots the distributions on a linear
#    scale. Set ``linear=0`` for log-log axes, which is the standard
#    way to visually assess power-law behavior.

# %%
# **Size distribution** — The total number of active channels
# summed over the duration of each avalanche, shown for different
# time bin widths.

fig_size, fig_cond, dict_results_size = plot_avalanche_distribution(
    input_data=raw,
    plot="size",
    threshold_method="std",
    threshold=2.5,
    time_bins=[1, 2, 3, 5, 10],
    hist_bins=50,
    linear=0,
)
plt.show()

# %%
# **Duration distribution** — The number of consecutive time bins
# in which at least one channel is active, shown for different
# time bin widths.

fig_durations, fig_cond, dict_results_durations = plot_avalanche_distribution(
    input_data=raw,
    plot="duration",
    threshold_method="std",
    threshold=2.5,
    time_bins=[1, 2, 3, 5, 10],
    hist_bins=50,
    linear=0,
)
plt.show()

# %%
# .. tip::
#    Both functions return:
#
#    - Two matplotlib ``Figure`` object for further customization
#    - A dictionary containing the raw distribution data for each
#      time binning, which you can use for statistical testing
#      or custom plotting.

# %%
# 4. Detect Avalanches
# ----------------------
# We configure the detection pipeline with:
#
# - **Fixed threshold** at 2.5 standard deviations
# - **Minimum duration** of 2 time bins
# - **Minimum size** of 0 (no constraint on the number of active channels)

avalanche_pipeline = AvalanchePipeline(
    z_scoring='by_channel',
    threshold_method='std',
    threshold=2.5,
    min_duration=2,
    min_size=0
)

aval_results = avalanche_pipeline.process(raw)

# %%
# 5. Inspect Avalanche Statistics
# ---------------------------------
# We look at the results and print basic summary statistics.

aval_data = aval_results.get_trial_data(0)

aval_count = aval_data.avalanche_count
durations = aval_data.durations
sizes = aval_data.sizes

print(f"Avalanches detected (epoch 0): {aval_count}")
print(f"Mean duration:  {np.mean(durations):.2f} bins")
print(f"Mean size:      {np.mean(sizes):.2f} active channels")
print(f"Avalanche data available in the object: {vars(aval_data).keys()}")

# %%
# 6. Compute Avalanche Transition Matrix (ATM)
# -----------------------------------------------
# The ATM captures how neural activity propagates between
# sensors during avalanches. We symmetrize the matrix so that
# :math:`ATM = (ATM + ATM^T) / 2`.

atm_pipeline = ATMPipeline(symm=True, method='standard')
atm_results = atm_pipeline.process(aval_results)

# %%
# 7. Visualize the Mean ATM
# ---------------------------
# Each cell ``(i, j)`` represents the average probability that
# channel *j* becomes active right after channel *i* during
# an avalanche.

atm_data = atm_results.get_trial_data(0)
mean_atm = atm_data.mean_atm

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    mean_atm,
    cmap='viridis',
    square=True,
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={'label': 'Transition Probability'},
)
ax.set_xlabel('Target Channel')
ax.set_ylabel('Source Channel')
ax.set_title('Mean Avalanche Transition Matrix')
plt.tight_layout()
plt.show()

# %%
# 8. Compute Network Metrics
# -----------------------------
# We extract graph-theoretic features from the ATM. These can
# be used for group comparisons or as input to a classifier.
#
# .. seealso::
#    :mod:`pyAvalanches.connectivity_metrics.NetworkMetricsPipeline` for the full list of
#    supported metrics.

metrics_pipeline = NetworkMetricsPipeline(
    metrics=['clustering_coeff', 'node_strength'],
    symm=True
)
metrics_results = metrics_pipeline.process(atm_results)

metrics_data = metrics_results.get_trial_data(0)

print(f"Average clustering coefficient: {float(metrics_data.avg_clustering_coefficient)}")

# %%
# 9. Save Subject Results
# --------------------------
# We can bundle everything into a
# :class:`~pyAvalanches.subject_group_results.Subject` object
# and save to disk.
# .. code-block:: python
#        subject = Subject(
#            subject_id='sub_brainstorm',
#            avalanche_results=aval_results,
#            atm_results=atm_results,
#            metrics_results=metrics_results,
#        )
#
#        subject.save(out_path='./results', prefix='openneuro_rest')
#
#        print("Results saved to ../results/")


# %%
# Generate an MNE Report
# ------------------------
# For a comprehensive visual summary of your results, you can
# generate an `MNE Report <https://mne.tools/stable/auto_tutorials/intro/70_report.html>`_
# with a single function call.
#
# The :func:`~pyAvalanches.output.generate_report` function
# automatically inspects the input object and adapts its behavior:
#
# - **Subject** → subject-level report
# - **GroupResults** → group-level report with per-subject panels
#
# The report includes (depending on available results):
#
# - **Summary tables** — Avalanche counts, mean size, mean duration
# - **Size distributions** — Log-log and KDE plots
# - **Duration distributions** — Log-log and KDE plots
# - **ATM heatmaps** — Per-epoch and mean transition matrices
#
# .. tip::
#
#    .. code-block:: python
#
#       from pyAvalanches.output import generate_report
#
#       generate_report(
#           subject=subject,
#           out_path=out_dir,
#           fname="openneuro_rest_report.html"
#           )

# %%
# The report is saved automatically; open it in your
# browser to explore the results interactively.
#
#
# .. tip::
#
#    You can also generate a **group-level** report by passing a
#    :class:`~pyAvalanches.subject_group_results.GroupResults` object:
#
#    .. code-block:: python
#
#       from pyAvalanches.subject_group_results import GroupResults
#
#       group = GroupResults(subjects=[subject_1, subject_2, subject_3])
#
#       generate_report(
#           results=group,
#           out_path='./results',
#           prefix='my_study',
#           overwrite=True,
#       )
#
#    This will produce ``./results/my_study_report.html`` containing
#    all subjects.