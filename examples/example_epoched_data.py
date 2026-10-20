"""
Epoched EEG — Motor Imagery Analysis
=======================================

This example demonstrates the full **pyAvalanches** pipeline on
epoched EEG data, using motor imagery epochs (hands vs feet) to
showcase condition-wise analysis.

We go through every step of the pipeline: avalanche detection,
transition matrices with **per-condition comparison**, network
metrics, visualization, report generation, and DataFrame export.

Dataset
-------

We use the **EEG Motor Movement/Imagery Dataset** from
`PhysioNet <https://physionet.org/content/eegmmidb/1.0.0/>`_,
available directly through :func:`mne.datasets.eegbci.load_data`.

.. seealso::

   - :doc:`/auto_examples/example_resting_state` — Resting-state
     single-subject analysis
   - :doc:`/auto_examples/example_classification` — Use ATM features
     in a scikit-learn classification pipeline
"""

# %%
# Imports
# -------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from pyAvalanches.dataset import create_dataset_from_mne, plot_amplitude_distribution
from pyAvalanches.avalanches import AvalanchePipeline
from pyAvalanches.transition_matrices import ATMPipeline
from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline
from pyAvalanches.subject_group_results import Subject
from pyAvalanches.output import (
    plot_subject_atm_heatmap,
    plot_subject_size_kde,
    plot_subject_duration_kde,
    generate_report,
    compute_summary_dataframes,
)

# %%
# 1. Load and Preprocess EEG Data
# ----------------------------------
# We load 3 motor imagery runs (hands vs feet) from subject 1
# of the PhysioNet dataset. The data is downloaded automatically
# on first run.

tmin, tmax = 1.0, 4.0  # Avoid evoked responses: start 1s after cue
subjects = 1
runs = [6, 10, 14]     # Motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subjects, runs, update_path=True)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

# Standardize channel names and set montage
eegbci.standardize(raw)
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)

# Rename annotations as documented on PhysioNet
raw.annotations.rename(dict(T1="hands", T2="feet"))
raw.set_eeg_reference(projection=True)

# Bandpass filter 7–30 Hz (mu and beta bands)
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")

# %%
# 2. Create Epochs
# ------------------
# We extract epochs from 1.0 to 4.0 s after cue onset. This
# avoids classifying the evoked response and captures sustained
# motor imagery activity across a longer window.

picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

epochs = Epochs(
    raw,
    event_id=["hands", "feet"],
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)

labels = epochs.events[:, -1] - 2

print(f"Number of epochs: {len(epochs)}")
print(f"Classes: hands={np.sum(labels == 2)}, feet={np.sum(labels == 3)}")

# %%
# 3. Z-Score and Inspect Amplitude Distribution
# -------------------------
# We z-score across the entire recording and inspect the amplitude
# distribution before binarizing.
# The amplitude distribution should be approximately Gaussian
# after z-scoring. This helps verify that the threshold will
# select meaningful activations.

plot_amplitude_distribution(epochs)

# %%
# 4. Detect Avalanches
# ----------------------
# We configure the detection pipeline with a standard-deviation
# based threshold and minimum duration of 2 time bins.

avalanche_pipeline = AvalanchePipeline(
    z_scoring='by_trial',
    threshold_method='std',
    threshold=2.5,
    min_duration=2,
    min_size=0,
)

aval_results = avalanche_pipeline.process(epochs)

# %%
# 5. Inspect Single-Epoch Results
# ----------------------------------
# We look at epoch 1 and print basic summary statistics. The
# result object contains all avalanche-level data for further
# analysis.

aval_data = aval_results.get_trial_data(1)

aval_count = aval_data.avalanche_count
durations = aval_data.durations
sizes = aval_data.sizes

print(f"Avalanches detected (epoch 1): {aval_count}")
print(f"Mean duration:  {np.mean(durations):.2f} bins")
print(f"Mean size:      {np.mean(sizes):.2f} active channels")
print(f"Available fields: {list(vars(aval_data).keys())}")

# %%
# 6. Compute Avalanche Transition Matrices
# -------------------------------------------
# The ATM captures sensor-to-sensor propagation patterns.
# With epoched data, we get one ATM per epoch — enabling
# condition-wise comparisons.

atm_pipeline = ATMPipeline(symm=True, method='standard')
atm_results = atm_pipeline.process(aval_results)

# %%
# 7. Compare ATMs Across Conditions
# ------------------------------------
# Since our epochs have condition labels (hands=2, feet=3),
# we can compute the **mean ATM per condition** and visualize
# differences in propagation dynamics.

mean_atm_hands = atm_results.get_atm_mean_by_condition(2)
mean_atm_feet = atm_results.get_atm_mean_by_condition(3)

# Use a shared color scale for fair comparison
vmin = min(mean_atm_hands.min(), mean_atm_feet.min())
vmax = max(mean_atm_hands.max(), mean_atm_feet.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(
    mean_atm_hands, cmap='viridis', ax=axes[0],
    vmin=vmin, vmax=vmax,
    xticklabels=False, yticklabels=False,
)
axes[0].set(xlabel="Channels", ylabel="Channels")
axes[0].set_title('Mean ATM — Hands Condition')

sns.heatmap(
    mean_atm_feet, cmap='viridis', ax=axes[1],
    vmin=vmin, vmax=vmax,
    xticklabels=False, yticklabels=False,
)
axes[1].set(xlabel="Channels", ylabel="Channels")
axes[1].set_title('Mean ATM — Feet Condition')

plt.tight_layout()
plt.show()

# %%
# 8. Compute Network Metrics
# ----------------------------
# We extract graph-theoretic features from each epoch's ATM.

metrics_pipeline = NetworkMetricsPipeline(
    metrics=['clustering_coeff', 'node_strength'],
    symm=True,
)
metrics_results = metrics_pipeline.process(atm_results)

metrics_data = metrics_results.get_trial_data(1)
print(f"Average clustering coefficient (epoch 1): "
      f"{float(metrics_data.avg_clustering_coefficient):.4f}")

# %%
# 9. Save Subject Results
# --------------------------
# We bundle everything into a
# :class:`~pyAvalanches.subject_group_results.Subject` object
# for storage and downstream analysis.

subject = Subject(
    subject_id='subj_eegbci',
    avalanche_results=aval_results,
    atm_results=atm_results,
    metrics_results=metrics_results,
)

subject.save(out_path='./results', prefix='subj_eegbci')
print("Results saved to ../results/")

# %%
# 10. Visualize with Built-In Plotting Functions
# -------------------------------------------------
# pyAvalanches provides a rich set of plotting functions that
# return matplotlib ``Figure`` objects for further customization.
#
# .. tip::
#
#    See :mod:`pyAvalanches.output` for the complete list of
#    available plotting functions.

plot_subject_size_kde(subject)

# %%

plot_subject_duration_kde(subject)

# %%

plot_subject_atm_heatmap(subject)

# %%
# 11. Generate an MNE Report
# ----------------------------
# Generate a comprehensive HTML report that includes all
# available results: summary tables, distributions, and
# ATM heatmaps.

#. tip::
#
#    .. code-block:: python
#
#       from pyAvalanches.output import generate_report
#
#       generate_report(
#           subject=subject,
#           out_path=out_dir,
#           fname="subj_eegbci_report.html",
#       )

# %%
# 12. Export Results to DataFrames
# ----------------------------------
# For statistical testing, group analysis, or integration with
# other tools, export all results to tidy pandas DataFrames.

df_aval, df_pattern, df_atm, df_metrics = compute_summary_dataframes(
    subject,
    ['avalanche', 'atm', 'metrics']
)

# %%

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
#    These DataFrames can be saved for later use:
#
#    .. code-block:: python
#
#       df_aval.to_csv('./results/avalanche_stats.csv', index=False)
#       df_metrics.to_csv('./results/network_metrics.csv', index=False)

# %%
# Summary
# -------
# In this example we:
#
# 1. Loaded a public motor imagery EEG dataset via MNE
# 2. Created epoched data with **condition labels** (hands vs feet)
# 3. Detected neuronal avalanches per epoch
# 4. Computed and compared ATMs **across conditions**
# 5. Extracted network metrics
# 6. Visualized results using built-in plotting functions
# 7. Generated an interactive HTML report
# 8. Exported tidy DataFrames for downstream analysis
#
# This workflow is the foundation for condition-wise comparisons
# in event-related paradigms, and the exported features can be
# fed directly into classification pipelines (see
# :doc:`/auto_examples/example_classification`).