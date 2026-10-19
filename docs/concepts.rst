.. _concepts:

====================================
Concepts & Package Structure
====================================

This page explains the core concepts and data structures in **pyAvalanches**.
After reading it, you should have a solid understanding of how the package
is organized and how to interact with its objects at every stage of the
analysis pipeline.

.. contents:: On this page
   :local:
   :depth: 2


Overview
========

A typical pyAvalanches workflow follows three stages, each producing its own
result object:

.. code-block:: text

        Avalanches        Avalanches Transition Matrices   Network Metrics
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │  TrialAvalanches  │ ──▸  │     TrialATM      │ ──▸  │    TrialMetric    │
   │ (AvalancheObj ×N) │      │    (ATMObj ×N)    │      │   (MetricObj ×N)  │
   └───────────────────┘      └───────────────────┘      └───────────────────┘
    Detected avalanches        Transition matrices        Computed metrics
    per trial                  per trial                  per trial

Each stage wraps its results in a **container** that holds one result object
per trial. This consistent trial-level structure is the backbone of the
package.


.. _trials_and_labels:

Trials and Labels
=================

Everything in pyAvalanches is organized around **trials**.

- A **trial** is a single, contiguous segment of neural data with shape
  ``(n_channels, n_time_points)``.
- Each trial has a **trial ID** (``trial_id``): a unique integer index
  (0, 1, 2, …).
- Each trial can optionally have a **trial label** (``trial_label``): a
  condition tag such as ``"task"``, ``"rest"``, ``1``, ``2``, etc.

.. list-table::
   :header-rows: 1
   :widths: 15 15 50

   * - Attribute
     - Type
     - Meaning
   * - ``trial_id``
     - ``int``
     - Unique positional index of the trial (auto-assigned)
   * - ``trial_label``
     - ``int`` or ``str``
     - Experimental condition (e.g. ``"task"``, ``2``). Optional.

Labels are propagated automatically through the pipeline: when you detect
avalanches or compute transition matrices, each result object inherits the
label of the trial it came from. This makes it straightforward to filter or
aggregate results by condition later on.

.. tip::
   You can retrieve all labels at any stage with the ``.trial_labels``
   property — it works on all objects :class:`~pyAvalanches.avalanches.TrialAvalanches`,
   :class:`~pyAvalanches.transition_matrices.TrialATM` and :class:`~pyAvalanches.connectivity_metrics.TrialMetrics`


.. _epoched_vs_resting:

Epoched vs. Resting-State Data
==============================

pyAvalanches handles both task-related (epoched) and resting-state
(continuous) recordings. The key difference is how trials are defined.

Epoched data comes pre-segmented into multiple trials, typically
time-locked to experimental events. Each epoch becomes one trial, and
each trial inherits the event label from the experimental paradigm.

Continuous recordings have no event-based segmentation. When you pass an
MNE ``Raw`` object (or a dictionary with a 2-D NumPy array), **the entire recording is
treated as a single trial**.

.. important::
   For resting-state data, the dataset contains **one trial with no
   label**. All downstream objects (``TrialAvalanches``, ``TrialATM``, ``TrialMetrics``)
   will therefore also contain a single entry. This is by design —
   the trial-based structure is preserved so that the same API works
   uniformly for both paradigms.

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * -
     - **Epoched**
     - **Resting-state**
   * - Input
     - MNE ``Epochs`` / 3-D array
     - MNE ``Raw`` / 2-D array
   * - ``n_trials``
     - Number of epochs
     - **1**
   * - ``trial_labels``
     - Event codes (e.g. ``[1, 2, 1, …]``)
     - None
   * - Data shape
     - ``(n_trials, n_channels, n_time_points)``
     - ``(1, n_channels, n_time_points)``


.. _avalanche_objects:

Avalanche Objects
=================

The results of avalanches detection are stored in two nested objects.

AvalancheObj — single trial
---------------------------

:class:`~pyAvalanches.avalanches.AvalancheObj` holds the avalanche detection
results for **one trial**. It contains everything you need to inspect the
avalanches found in that segment of data.

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Attribute
     - Description
   * - ``trial_id``
     - Unique trial index
   * - ``trial_label``
     - Condition label (inherited from the dataset)
   * - ``avalanche_count``
     - Total number of detected avalanches
   * - ``sizes``
     - List of avalanche sizes (number of activations)
   * - ``durations``
     - List of avalanche durations (in time bins)
   * - ``avalanches_mat``
     - List of binarized spatiotemporal matrices, one per avalanche
   * - ``patterns``
     - Spatial activation patterns per avalanche
   * - ``patterns_temporal``
     - Temporal activation patterns per avalanche
   * - ``bounds``
     - ``(start, end)`` time-point indices per avalanche
   * - ``flexibility``
     - Number of distinct spatial patterns across avalanches

.. code-block:: python

   >>> trial_result
   AvalancheObj(trial_id=0, n_avalanches=34)
   >>> trial_result.sizes
   [5, 12, 3, 8, ...]
   >>> trial_result.durations
   [2, 4, 1, 3, ...]

TrialAvalanches — all trials
----------------------------

:class:`~pyAvalanches.avalanches.TrialAvalanches` is the **container** that
collects one :class:`~pyAvalanches.avalanches.AvalancheObj` per trial.
This is the object returned by the detection pipeline and the main one you
will interact with.

.. code-block:: python

   >>> avalanches
   TrialAvalanches(n_trials=40)

**Accessing individual trials:**

.. code-block:: python

   # By trial ID
   trial_0 = avalanches.get_trial_data(trial_idx=0)
   print(trial_0.avalanche_count)  # 34

   # Iterate over all trials
   for trial in avalanches.trials:
       print(trial.trial_id, trial.trial_label, trial.avalanche_count)

**Retrieving results across trials:**

.. code-block:: python

   # All durations, merged into a single flat list
   all_durations = avalanches.get_all_durations(merged=True)

   # All durations, nested per trial
   durations_per_trial = avalanches.get_all_durations(merged=False)

   # Durations for a specific condition only
   rest_durations = avalanches.get_durations_by_cond("rest", merged=True)

   # Same interface for sizes
   all_sizes = avalanches.get_all_sizes(merged=True)
   task_sizes = avalanches.get_sizes_by_cond("task", merged=True)

**Useful properties:**

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Property / Method
     - Description
   * - ``trial_count``
     - Number of trials
   * - ``trial_labels``
     - List of all condition labels
   * - ``trial_indexes``
     - List of all trial IDs
   * - ``get_all_sizes(merged)``
     - Avalanche sizes across all trials
   * - ``get_all_durations(merged)``
     - Avalanche durations across all trials
   * - ``get_sizes_by_cond(cond, merged)``
     - Sizes filtered by condition
   * - ``get_durations_by_cond(cond, merged)``
     - Durations filtered by condition
   * - ``save(out_path, prefix)``
     - Serialize to pickle file

.. tip::
   The ``merged`` parameter controls the output format. Use
   ``merged=True`` for pooled distributions (e.g. for plotting
   histograms) and ``merged=False`` when you need per-trial
   granularity (e.g. for statistical tests).


.. _atm_objects:

Transition Matrix Objects
=========================

After computing Avalanche Transition Matrices (ATMs) from the detected
avalanches, results are again stored in a trial-level structure that
mirrors :class:`~pyAvalanches.avalanches.TrialAvalanches`.

ATMObj — single trial
---------------------

:class:`~pyAvalanches.transition_matrices.ATMObj` holds the ATM
results for **one trial**.

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Attribute
     - Description
   * - ``trial_id``
     - Unique trial index
   * - ``trial_label``
     - Condition label (inherited)
   * - ``avalanche_transition_matrices``
     - List of ATMs, one per avalanche
   * - ``mean_atm``
     - Mean ATM averaged across all avalanches in the trial
   * - ``symmetric``
     - Whether the ATMs have been symmetrized
   * - ``method``
     - Method used to compute the ATM

TrialATM — all trials
---------------------

:class:`~pyAvalanches.transition_matrices.TrialATM` is the **container**
collecting one :class:`~pyAvalanches.transition_matrices.ATMObj` per trial.

.. code-block:: python

   >>> atm_results
   TrialATMs(n_trials=40)

**Accessing individual trials:**

.. code-block:: python

   trial_0_atm = atm_results.get_trial_data(trial_idx=0)
   print(trial_0_atm.mean_atm.shape)  # (n_channels, n_channels)

**Computing mean ATMs:**

.. code-block:: python

   # Global mean ATM across all trials
   global_atm = atm_results.global_mean_atm()

   # Mean ATM for a specific condition
   rest_atm = atm_results.get_atm_mean_by_condition("rest")
   task_atm = atm_results.get_atm_mean_by_condition("task")

**Useful properties and methods:**

.. list-table::
   :header-rows: 1
   :widths: 35 50

   * - Property / Method
     - Description
   * - ``trial_count``
     - Number of trials
   * - ``trial_labels``
     - List of all condition labels
   * - ``trial_indexes``
     - List of all trial IDs
   * - ``global_mean_atm()``
     - Mean ATM across all trials
   * - ``get_atm_mean_by_condition(cond)``
     - Mean ATM filtered by condition
   * - ``save(out_path, prefix)``
     - Serialize to pickle file

.. _metrics_objects:

Network Metrics Objects
=========================

Network metrics can be computed from Avalanche Transition Matrices (ATMs)
and results are stored in a trial-level structure like
:class:`~pyAvalanches.avalanches.TrialAvalanches` and
:class:`~pyAvalanches.transition_matrices.TrialATM`.

MetricsObj — single trial
---------------------

:class:`~pyAvalanches.connectivity_metrics.MetricsObj` holds the metrics
results for **one trial**.

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Attribute
     - Description
   * - ``trial_id``
     - Unique trial index
   * - ``trial_label``
     - Condition label (inherited)
   * - ``atm_method``
     - Method used to compute the ATM
   * - ``metric_name``
     - An attribute is generated for each computed metric (user choice)

.. seealso::

   :mod:`pyAvalanches.connectivity_metrics.NetworkMetricsPipeline` for the full list of
   supported metrics.

TrialMetric — all trials
---------------------

:class:`~pyAvalanches.connectivity_metrics.TrialMetrics` is the **container**
collecting one :class:`~pyAvalanches.connectivity_metrics.MetricObj` per trial.


.. code-block:: python

   >>> metrics_results
   TrialMetrics(n_trials=40)

**Accessing individual trials:**

.. code-block:: python

   trial_0_metrics = metrics_results.get_trial_data(trial_idx=0)
   print(trial_0_atm.avg_node_strength)  # Average Node Strength

**Useful properties and methods:**

.. list-table::
   :header-rows: 1
   :widths: 35 50

   * - Property / Method
     - Description
   * - ``trial_count``
     - Number of trials
   * - ``trial_labels``
     - List of all condition labels
   * - ``trial_indexes``
     - List of all trial IDs
   * - ``save(out_path, prefix)``
     - Serialize to pickle file

.. _pipelines:

Pipelines
=========

pyAvalanches provides three pipeline classes that compute the detection
of the avalanches, the ATMs and the network metrics.
Each pipeline takes the output of the previous stage and produces the corresponding result object.

.. code-block:: text

   ┌───────────┐  AvalanchePipeline  ┌──────────────────┐  ATMPipeline  ┌────────────┐  NetworkMetricsPipeline   ┌────────────────┐
   │   data    │ ────────────────▸   │ TrialAvalanches  │ ───────────▸  │  TrialATM  │        ───────────▸       │  TrialMetrics  │
   └───────────┘      .process()     └──────────────────┘  .process()   └────────────┘         .process()        └────────────────┘

AvalanchePipeline
-----------------

:class:`~pyAvalanches.avalanches.AvalanchePipeline` detects neuronal
avalanches from a neural recording. It handles
thresholding, binarization, and avalanche identification for every trial.

**Key parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Parameter
     - Description
   * - ``z_scoring``
     - String or boolean (``"by_trial"``, ``"by_channel"`` or ``False``), indicates if the z-scoring should be done by each trial independently, by each channel independently or not
   * - ``threshold_method``
     - ``"fixed"`` (absolute value) or ``"std"`` (standard-deviation based)
   * - ``threshold``
     - Numeric threshold value (interpretation depends on method)
   * - ``min_duration``
     - Minimum avalanche duration in time bins (default: 1)
   * - ``min_size``
     - Minimum number of activations per avalanche (default: 1)
   * - ``bins``
     - Number of time bins for temporal subsampling (default: 1)
   * - ``trial_label``
     - If set, only trials with this label are processed

**Usage:**

:class:`~pyAvalanches.pipeline.AvalanchePipeline` accepts your neural time
series directly.
You can hand the pipeline either an MNE object or a plain dictionary holding a NumPy
array plus the metadata the pipeline needs.

**Accepted inputs**

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Input type
     - Notes
   * - :class:`mne.Epochs`
     - Trial-wise data; metadata is read from ``info``
   * - :class:`mne.io.Raw`
     - Continuous / resting-state data, treated as a single trial
   * - :class:`dict`
     - NumPy data array plus explicit metadata


When you pass an MNE object, sampling frequency, channel names and filter band
are taken from ``info``, so nothing else needs to be specified:

.. code-block:: python

   import mne
   from pyAvalanches.pipeline import AvalanchePipeline

   # From Epochs
   epochs = mne.read_epochs("sub-01-epo.fif")
   pipeline = AvalanchePipeline(
       zscore_by_trial = True,
       threshold_method="fixed",
       threshold=3.0,
       min_duration=2,
       min_size=3,
       bins=1,)
   results = pipeline.process(epochs)

   # From Raw (resting-state)
   raw = mne.io.read_raw_fif("sub-01-raw.fif", preload=True)
   results = pipeline.process(raw)

.. note::
   Make sure the data are loaded in memory (``preload=True``) and that the
   band-pass limits are set on the object (``raw.filter(...)`` /
   ``epochs.info["highpass"]`` and ``["lowpass"]``), since the pipeline reads
   the filter band from ``info``.


If your data do not come from MNE, wrap the array in a dictionary together with
the required metadata:

.. list-table::
   :header-rows: 1
   :widths: 25 15 50

   * - Key
     - Required
     - Description
   * - ``data``
     - yes
     - Array of shape ``(n_trials, n_channels, n_time_points)``
   * - ``samp_freq``
     - yes
     - Sampling frequency in Hz
   * - ``fmin`` / ``fmax``
     - yes
     - Filter band limits in Hz
   * - ``channel_names``
     - no
     - List of channel names or ROIs
   * - ``trial_labels``
     - no
     - List of condition labels, one per trial
   * - ``trial_ids``
     - no
     - List of unique trial identifiers
   * - ``t_max``
     - no
     - Maximum time points to import from the trial

.. warning::
   Even if some attributes are not required, they need to be present in the dictionary. Please
   set the arguments you are not providing as ``None``.

.. note::
   If you pass a 2-D array ``(n_channels, n_time_points)``, it will
   be automatically expanded to ``(1, n_channels, n_time_points)`` —
   i.e., a single trial, same as resting-state.

.. code-block:: python

   import numpy as np
   from pyAvalanches.pipeline import AvalanchePipeline

   # data shape: (n_trials, n_channels, n_time_points)
   data = np.random.randn(40, 64, 500)

   input_data = {
       "data": data,
       "samp_freq": 250.0,
       "fmin": 1.0,
       "fmax": 45.0,
       "channel_names": [f"Ch{i}" for i in range(64)],
       "trial_labels": ["task"] * 20 + ["rest"] * 20,
       "t_max": None
   }

   pipeline = AvalanchePipeline(
       zscore_by_trial = True,
       threshold_method="std",
       threshold=3.0,
       min_duration=2,
       min_size=3,
       bins=1)

   results = pipeline.process(input_data)


If you provided labels for each trial, you can process only a specific condition:

.. code-block:: python

   av_pipe_task = AvalanchePipeline(
       zscore_by_trial = True,
       threshold_method="std",
       threshold=3.0,
       trial_label="task",
   )

   avalanches_task = av_pipe_task.process(input_data)
   print(avalanches_task.trial_count)  # only "left" trials

ATMPipeline
-----------

:class:`~pyAvalanches.transition_matrices.ATMPipeline` computes Avalanche
Transition Matrices from a :class:`~pyAvalanches.avalanches.TrialAvalanches`
object.

Avalanche Transition Matrices quantify how likely neuronal avalanches propagate
between pairs of brain regions. Two computation methods are available:

- **``standard``** — element *(i, j)* is the probability that region *j*
  was active at time *t+1* given that region *i* was active at time *t* [1]_.
- **``ws``** (weighted-stochastic) — explicitely incorporates transition probabilities
  togheter with regional activation periods and avalanche durations [2]_.:

  1. **Transition counting** — count consecutive co-activations between
     each pair of regions.
  2. **Probability normalization** — normalize counts into probabilities.
  3. **Duration weighting** — weight each avalanche's ATM by its duration.

**Key parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Parameter
     - Description
   * - ``method``
     - ``"standard"`` or ``"ws"`` (weighted-stochastic)
   * - ``symm``
     - If ``True``, the resulting ATMs are symmetrized
   * - ``trial_label``
     - If set, only trials with this label are processed

.. warning::
   The ``symm`` parameter is **ignored** when using the ``"ws"`` method.
   Weighted-stochastic ATMs are inherently asymmetric and will always be
   returned without symmetrization.

.. rubric:: References

.. [1] Sorrentino, P., Seguin, C., Rucco, R., Liparoti, M., Troisi Lopez, E., Bonavita, S., ... & Zalesky, A.
   (2021). The structural connectome constrains fast brain dynamics. *Elife*, 10, e67400.
   `doi:10.7554/eLife.67400 <https://doi.org/10.7554/eLife.67400>`_

.. [2] Mannino, C., Sorrentino, P., Angiolelli, M., Demuru, M., Trojsi, F.,
   Chavez, M., & Corsi, M. C. (2026). Weighted-stochastic Avalanche
   Transition Matrix (ws-ATM): a tool to investigate brain dynamic and its
   neuropathological alterations. *medRxiv*, 2026-01.
   `doi:10.64898/2026.01.08.26343631 <https://doi.org/10.64898/2026.01.08.26343631>`_

**Usage:**

.. code-block:: python

   from pyAvalanches.transition_matrices import ATMPipeline

   atm_pipe = ATMPipeline(
       method="standard",
       symm=True,
   )

   atm_results = atm_pipe.process(avalanches)  # returns TrialATM
   print(atm_results)  # TrialATMs(n_trials=40)

   # Mean ATM across all trials
   global_atm = atm_results.global_mean_atm()

   # Mean ATM for a specific condition
   left_atm = atm_results.get_atm_mean_by_condition("left")

NetworkMetricsPipeline
-----------------------

:class:`~pyAvalanches.connectivity_measures.NetworkMetricsPipeline` computes network metrics
from a :class:`~pyAvalanches.transition_matrices.TrialATM` object.

.. important::
   This pipeline is mostly a wrapper calling network metrics computing functions from
   `Brain Connectivity Toolbox <https://pypi.org/project/bctpy/>`_ [3]_
   and `NetworkX <https://networkx.org/en/>`_ [4]_. Check out the respective documentation
   to know more about the functions.


**Key parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Parameter
     - Description
   * - ``metrics``
     - List of metrics to compute
   * - ``symm``
     - If the pipeline should expect symmetric or non-symmetric ATMs

Available metrics include:

- ``clustering_coeff`` — Clustering Coefficient
- ``node_strength`` — Node Strenght
- ``eigenvect_centrality`` - Eigenvector Centrality
- ``modularity`` - Modularity
- ``char_path`` - Characteristic Path Length
- ``connectivity_measures`` - mean, variance, coefficient of variation, skewness

.. seealso::

   :mod:`pyAvalanches.connectivity_metrics.NetworkMetricsPipeline` for the full list of
   supported metrics and how to input them.

.. rubric:: References

.. [3] Rubinov, M., Kötter, R., Hagmann, P., & Sporns, P. (2009). Brain connectivity toolbox:
   a collection of complex network measurements and brain connectivity datasets. *NeuroImage*, 47(1).
   `doi:10.1016/S1053-8119(09)71822-1 <https://doi.org/10.1016/S1053-8119(09)71822-1>`_

.. [4] Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics,
   and function using NetworkX. *Proceedings of the 7th Python in Science Conference (SciPy2008)*, 11–15.
   `http://conference.scipy.org.s3-website-us-east-1.amazonaws.com/proceedings/scipy2008/paper_2/
   <http://conference.scipy.org.s3-website-us-east-1.amazonaws.com/proceedings/scipy2008/paper_2/>`_

**Usage:**

.. code-block:: python

   from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline

   metrics_pipeline = NetworkMetricsPipeline(
       metrics=['clustering_coeff', 'node_strength'],
       symm=True,
   )

   metrics_results = metrics_pipeline.process(atm_results)  # returns TrialATM
   print(metrics_results)  # TrialMetrics(n_trials=40)


.. _grouping_results:

Grouping Results
================

You might want to store the results you obtained from the pipelines by subjects or groups of subjects.
:class:`~pyAvalanches.subject_group_results_new.Subject` and :class:`~pyAvalanches.subject_group_results_new.GroupResults`
are designed to easily pull these results together and access them.

Subject
--------

:class:`~pyAvalanches.subject_group_results_new.Subject` can store data information,
avalanche results, ATM and metrics, identifying the subject by an ID and a label (for example
"control" or "patient") if needed to group them later. Every result is optional to store.

**Key parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Attribute
     - Description
   * - ``subject_id``
     - Unique subject identifier
   * - ``subject_label``
     - Group label
   * - ``dataset``
     - Stores all dataset information, except for the neural recording themself
   * - ``avalanche_results``
     - Avalanche results (TrialAvalanches object)
   * - ``atm_results``
     - ATM results (TrialATM object)
   * - ``metrics_results``
     - Network metrics results (TrialMetrics object)

.. code-block:: python

   from pyAvalanches.subject_group_results_new import Subject

   subj_01 = Subject(
        subject_id=1,
        subject_label="control",
        dataset=epochs,
        avalanche_results=aval_results,
        atm_results=atm_results,
        metrics_results=metrics_results
   )

   avalanches_task = av_pipe_task.process(input_data)
   print(avalanches_task.trial_count)  # only "left" trials

The Subject object allows to quickly access subject-level results, especially if
you are working with multiple trials:

**Useful properties and methods:**

.. list-table::
   :header-rows: 1
   :widths: 35 50

   * - Property / Method
     - Description
   * - ``get_mean_atm_cond(cond)``
     - Mean ATM across all trials matching a condition
   * - ``get_summary_dataframe()``
     - Builds a DataFrame of this subject's analysis results
   * - ``save(out_path, prefix)``
     - Serialize to pickle file and saves the summary dataframe

GroupResults
-------------

Following the same structure of the results objects, :class:`~pyAvalanches.subject_group_results_new.GroupResults`
is the **container** for collecting :class:`~pyAvalanches.subject_group_results_new.Subject`.

.. code-block:: python

   from pyAvalanches.subject_group_results_new import Subject, GroupResults

   subj_01 = Subject(
        subject_id=1,
        subject_label="control",
        dataset=epochs,
        avalanche_results=aval_results,
        atm_results=atm_results,
        metrics_results=metrics_results
   )

   subj_02 = Subject(
        subject_id=2,
        subject_label="patient",
        dataset=epochs,
        avalanche_results=aval_results,
        atm_results=atm_results,
        metrics_results=metrics_results
   )

   group_results = GroupResults(subjects=[subj_01, subj_02])

It provides useful methods to extract results at a group-level.

.. list-table::
   :header-rows: 1
   :widths: 35 50

   * - Property / Method
     - Description
   * - ``add_subject(subject)``
     - Add a new Subject to GroupResults
   * - ``get_subject(subject_id)``
     - Retrieve a specific Subject by its ID
   * - ``get_avalanche_subj(subject_id)``
     - Retrieve avalanches results of a Subject by its ID
   * - ``get_atm_subj(subject_id)``
     - Retrieve ATM results of a Subject by its ID
   * - ``get_metrics_subj(subject_id)``
     - Retrieve metrics results of a Subject by its ID
   * - ``get_mean_subj_atm(subject_id)``
     - Mean ATM of a specific Subject
   * - ``get_mean_atm_group(group, condition)``
     - Mean ATM across all subjects with a given group label, for one condition.
   * - ``get_summary_dataframe()``
     - Builds a DataFrame of analysis results
   * - ``save(out_path, prefix)``
     - Serialize to pickle file

.. _full_example:

Putting It All Together
=======================

Here is a minimal end-to-end example showing how the objects flow through
the pipeline:

.. tip::
   See :doc:`/usage` for the detailed steps of the workflow,
   setting up the pipelines to compute Avalanches and ATMs, and store results
   on subject or group level.

..  code-block:: python

    import mne

    # 1. Loading data

    n_trials = 3
    n_channels = 4
    n_timepoints = 512

    data = np.random.randn(n_trials, n_channels, n_timepoints)

    input_data = {
        "data_array": data,
        "samp_freq": 256.0,
        "fmin": 1.0,
        "fmax": 40.0,
        "channel_names": ["Fp1", "Fp2", "C3", "C4"],
        "trial_labels": ["left_hand", "right_hand", "left_hand"],
        "t_max": None
    }

    # 2. Detecting avalanches

    avalanche_pipeline = AvalanchePipeline(
        z_score_by_trial=True,
        threshold_method='std',
        threshold=1.5,
        min_duration=2,
        min_size=2,)

    aval_results = avalanche_pipeline.process(input_data)

    # 3. Computing ATMs

    atm_pipeline = ATMPipeline(
        method='standard',
        symm=True,
    )

    atm_results = atm_pipeline.process(aval_results)

    # 4. Computing network metrics

    metrics_pipeline = NetworkMetricsPipeline(metrics=["node_strength","eigenvect_centrality","connectivity_measures"], symm=True)
    metrics_results = metrics_pipeline.process(atm_results)

    # 5. Grouping results

    subj_0 = Subject(
        subject_id=0,
        subject_label="SM",
        dataset=input_data,
        avalanche_results=aval_results,
        atm_results=atm_results,
        metrics_results=metrics_results
    )

    group_results = GroupResults(subjects=[subj_0])

For resting-state data the flow is identical — the only difference is that
``TrialAvalanches``, ``TrialATM`` and ``TrialMetrics`` will each contain a
**single entry** with no condition label.


.. _object_summary:

Object Summary
==============

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - Stage
     - Object
     - Contains
   * - Avalanche Detection (per trial)
     - :class:`~pyAvalanches.avalanches.AvalancheObj`
     - Sizes, durations, patterns for one trial
   * - Avalanche Detection (all trials)
     - :class:`~pyAvalanches.avalanches.TrialAvalanches`
     - Collection of ``AvalancheObj``
   * - ATM (per trial)
     - :class:`~pyAvalanches.transition_matrices.ATMObj`
     - Transition matrices for one trial
   * - ATM (all trials)
     - :class:`~pyAvalanches.transition_matrices.TrialATM`
     - Collection of ``ATMObj``
   * - Network Metrics (per trial)
     - :class:`~pyAvalanches.connectivity_metrics.MetricsObj`
     - Network metrics for one trial
   * - Network Metrics (all trials)
     - :class:`~pyAvalanches.connectivity_metrics.TrialMetrics`
     - Collection of ``MetricsObj``
   * - Subject (per subject results)
     - :class:`~pyAvalanches.subject_group_results_new.Subject`
     - Transition matrices for one trial
   * - GroupResults (all subjects)
     - :class:`~pyAvalanches.subject_group_results_new.GroupResults`
     - Collection of ``Subject``