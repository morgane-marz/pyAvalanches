Usage
=====

.. _installation:

Installation
------------

To use pyAvalanches, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pyAvalanches

Basic Workflow
--------------

This guide walks you through the full **pyAvalanches** pipeline, from
loading neural recordings to extracting network-level features.

.. contents:: Steps
   :local:
   :depth: 1

----

1. Load Your Data
-----------------

pyAvalanches accepts `MNE-Python <https://mne.tools>`_ objects or custom 2D and 3D numpy arrays of data wrapped
in a dictionary with metadata.

From MNE Epochs
~~~~~~~~~~~~~~~

If you are working with `MNE-Python <https://mne.tools>`_ ``Epochs`` and ``Raw``
objects, create them and they will be ready to use by the pipelines:

.. code-block:: python

   import mne
   # epoched data
   epochs = mne.read_epochs("sub-01-epo.fif")

   # raw data
   raw = mne.io.read_raw_fif("sub-01-raw.fif", preload=True)

.. tip::
   If your data is stored in BIDS format, check out how to load them with
   `MNE-BIDS <https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html>`_ .

From a NumPy Array
~~~~~~~~~~~~~~~~~~

You can also input your data from a NumPy array. You will need to provide them and
to specify the recording parameters manually in a dictionary:

.. code-block:: python

   import numpy as np

   n_trials = 3
   n_channels = 4
   n_timepoints = 512

   data = np.random.randn(n_trials, n_channels, n_timepoints)

   example_dict = {
        "data_array": data,      # your 2D or 3D data array
        "samp_freq": 256.0,      # sampling frequency in Hz
        "fmin": 1.0,             # lower frequency bound (Hz)
        "fmax": 40.0,            # upper frequency bound (Hz)
        "channel_names": ["Fp1", "Fp2", "C3", "C4"],
        "trial_labels": ["left", "right", "left"],
        "t_max": None
   }

.. note::

   ``data_array`` should have shape ``(n_epochs, n_channels, n_times)``
   for epoched data, or ``(n_channels, n_times)`` for continuous
   recordings.

.. warning::

   No filtering is applied by default. The ``fmin`` and ``fmax``
   parameters in the constructor are **metadata only** — they do
   not trigger any filtering.

2. Detect Avalanches
--------------------

Set up ``AvalanchePipeline`` with all the parameters to compute avalanches.
Before detection, the data is z-scored to normalize activity across channels.

.. code-block:: python

   from pyAvalanches.avalanches import AvalanchePipeline

   # Configure and run the detection pipeline
   avalanche_pipeline = AvalanchePipeline(
        z_scoring='by_trial',
        threshold_method='std',
        threshold=1.5,
        min_duration=2,
        min_size=2
        )

   avalanche_results = aval_pipeline.process(example_dict)

.. list-table:: AvalanchePipeline Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``zscore_by_trial``
     - str or bool
     - If ``'by_trial'``, z-scoring is applied independently to each trial;
       if ``'std'``, z-scoring is applied per channel across all trials and time points combined;
       if ``False``, don't z-score the data
   * - ``threshold_method``
     - str
     - Method for thresholding. ``'std'`` applies a threshold based
       on the signal's standard deviation;
       ``'fixed'`` applies a fixed threshold value.)
   * - ``threshold``
     - float
     - Threshold value, interpreted according to ``threshold_method``
       (e.g. ``3`` for 3 standard deviations)
   * - ``min_duration``
     - int, optional
     - Minimum duration in time bins for a valid avalanche
   * - ``min_size``
     - int, optional
     - Minimum number of activations for a valid avalanche
   * - ``bins``
     - int, optional
     - Number of time bins used for time subsampling
   * - ``trial_label``
     - int or str, optional
     - If specified, only trials with this label are processed

.. seealso::
   For a complete overview of the behavior of ``AvalanchePipeline`` and its results
   check out :doc:`Concept and Package Structure <concepts>`

----

3. Compute Avalanche Transition Matrices
-----------------------------------------

The **Avalanche Transition Matrix (ATM)** quantifies how likely is for a neuronal avalanche
to propagate between pairs of channels or brain regions.

Two computation methods are available:

- ``'standard'`` — element *(i, j)* is the probability that region *j*
  was active at time *t+1* given that region *i* was active at time *t* [1]_.
- ``'ws'`` (weighted-stochastic) — explicitely incorporates transition probabilities
  togheter with regional activation periods and avalanche durations [2]_.:

.. code-block:: python

   from pyAvalanches.transition_matrices import ATMPipeline

   atm_pipeline = ATMPipeline(
       method='standard',
       symm=True
   )
   atm_results = atm_pipeline.process(avalanche_results)

.. list-table:: ATM Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``method``
     - str
     - ATM computation method (``'standard'`` or ``ws``)
   * - ``symm``
     - bool
     - If ``True``, symmetrize the matrix: :math:`ATM = (ATM + ATM^T) / 2`

.. warning::
   The ``symm`` parameter is **ignored** when using the ``"ws"`` method.
   Weighted-stochastic ATMs are inherently asymmetric and will always be
   returned without symmetrization.

.. seealso::
   For a complete overview of the behavior of ``ATMPipeline`` and its results
   check out :doc:`Concept and Package Structure <concepts>`


.. rubric:: References

.. [1] Sorrentino, P., Seguin, C., Rucco, R., Liparoti, M., Troisi Lopez, E., Bonavita, S., ... & Zalesky, A.
   (2021). The structural connectome constrains fast brain dynamics. *Elife*, 10, e67400.
   `doi:10.7554/eLife.67400 <https://doi.org/10.7554/eLife.67400>`_

.. [2] Mannino, C., Sorrentino, P., Angiolelli, M., Demuru, M., Trojsi, F.,
   Chavez, M., & Corsi, M. C. (2026). Weighted-stochastic Avalanche
   Transition Matrix (ws-ATM): a tool to investigate brain dynamic and its
   neuropathological alterations. *medRxiv*, 2026-01.
   `doi:10.64898/2026.01.08.26343631 <https://doi.org/10.64898/2026.01.08.26343631>`_

----

4. Compute Network Metrics
--------------------------

Extract graph-theoretic features from the transition matrices for
further statistical analysis or classification.

.. code-block:: python

   from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline

   metrics_pipeline = NetworkMetricsPipeline(
       metrics=['clustering_coeff', 'node_strength'],
       symm=True
   )
   metrics_results = metrics_pipeline.process(atm_results)

Available metrics include:

- ``clustering_coeff`` - Clustering Coefficient
- ``node_strength`` - Node Strenght
- ``eigenvect_centrality`` - Eigenvector Centrality
- ``modularity`` - Modularity
- ``char_path`` - Characteristic Path Length
- ``connectivity_measures`` - mean, variance, coefficient of variation, skewness

.. important::
   This pipeline is mostly a wrapper calling network metrics computing functions from
   `Brain Connectivity Toolbox <https://pypi.org/project/bctpy/>`_
   and `NetworkX <https://networkx.org/en/>`_ . Check out the respective documentation
   to know more about the functions and metrics.

.. warning::
   These network metrics are **computed on the Avalanche Transition Matrix (ATM)**,
   a directed graph where edge weights represent **conditional probabilities** of
   propagation. Therefore, **their interpretation should be adapted to this context**
   of directed information flow, which differs from metrics computed on static
   functional connectivity matrices.

.. seealso::
   For a complete overview of the behavior of ``NetworkMetricsPipeline`` and its results
   check out :doc:`Concept and Package Structure <concepts>`

----

5. Store and Save Results
-------------------------

Bundle all results into a :class:`~pyAvalanches.subject_group_results.Subject`
object for easy storage and export.

.. code-block:: python

  from pyAvalanches.subject_group_results import Subject

   subject = Subject(
       subject_id=1,
       subject_label='patient'
       avalanche_results=avalanche_results,
       atm_results=atm_results,
       metrics_results=metrics_results
   )

   subject.save(out_path='./results', prefix='sub_01')

This will write the results to disk under ``./results/sub_01_*``.

----

Full Pipeline
-------------

Here is the complete workflow in a single script:

.. code-block:: python

   from pyAvalanches.dataset import create_dataset_from_mne
   from pyAvalanches.avalanches import AvalanchePipeline
   from pyAvalanches.transition_matrices import ATMPipeline
   from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline
   from pyAvalanches.subject_group_results import Subject

   # 1. Load data
   n_trials = 3
   n_channels = 4
   n_timepoints = 512

   data = np.random.randn(n_trials, n_channels, n_timepoints)

   input_data = {
        "data_array": data,      # your 2D or 3D data array
        "samp_freq": 256.0,      # sampling frequency in Hz
        "fmin": 1.0,             # lower frequency bound (Hz)
        "fmax": 40.0,            # upper frequency bound (Hz)
        "channel_names": ["Fp1", "Fp2", "C3", "C4"],
        "trial_labels": ["left", "right", "left"],
        "t_max": None
   }

   # or MNE objects

   input_data = mne.read_epochs("sub-01-epo.fif")

   # 2. Detect avalanches
   aval_pipeline = AvalanchePipeline(
        z_score_by_trial=True,
        threshold_method='std',
        threshold=1.5,
        min_duration=2,
        min_size=2
        )
   avalanche_results = aval_pipeline.process(input_data)

   # 3. Compute transition matrices
   atm_pipeline = ATMPipeline(method='standard', symm=True)
   atm_results = atm_pipeline.process(avalanche_results)

   # 4. Extract network metrics
   metrics_pipeline = NetworkMetricsPipeline(
       metrics=['clustering_coeff', 'node_strength'],
       symm=True,
   )
   metrics_results = metrics_pipeline.process(atm_results)

   # 5. Save
   subject = Subject(
       subject_id=1,
       avalanche_results=avalanche_results,
       atm_results=atm_results,
       metrics_results=metrics_results,
   )
   subject.save(out_path='./results', prefix='sub_01')

.. tip::

   You can loop over multiple subjects and aggregate results using
   :class:`~pyAvalanches.subject_group_results_new.GroupResults` for
   group-level statistical comparisons.

.. seealso::
   For more details and functionality (there are many more!),
   check out the :doc:`Example Gallery <sphinx_gallery.gen_gallery>`