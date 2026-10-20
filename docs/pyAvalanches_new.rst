.. _api_ref:

=============
API Reference
=============

This section documents all public modules, classes, and functions
in **pyAvalanches**.

.. contents:: Module Overview
   :local:
   :depth: 1

.. _avalanches_ref:

:mod:`pyAvalanches.avalanches` -- Avalanche Detection
=====================================================

.. automodule:: pyAvalanches.avalanches
   :no-members:
   :no-inherited-members:

The main module for detecting and characterizing neuronal avalanches
from neural recordings.

Classes
-------

.. currentmodule:: pyAvalanches

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   avalanches.AvalancheObj
   avalanches.TrialAvalanches
   avalanches.AvalanchePipeline


Functions
---------

.. currentmodule:: pyAvalanches

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   avalanches.plot_avalanche_distribution


.. _transition_matrices_ref:

:mod:`pyAvalanches.transition_matrices` -- Transition Matrices
==============================================================

.. automodule:: pyAvalanches.transition_matrices
   :no-members:
   :no-inherited-members:

Compute **Avalanche Transition Matrices (ATM)** capturing
sensor-to-sensor or ROI-to-ROI propagation probabilities.

Classes
-------

.. currentmodule:: pyAvalanches

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   transition_matrices.ATMObj
   transition_matrices.TrialATM
   transition_matrices.ATMPipeline


.. _connectivity_metrics_ref:

:mod:`pyAvalanches.connectivity_metrics` -- Connectivity & Graph Metrics
========================================================================

.. automodule:: pyAvalanches.connectivity_metrics
   :no-members:
   :no-inherited-members:

Derive connectivity and graph-theoretic metrics from avalanche transition
matrices.

.. Note::
   The connectivity and graph-theoretic metrics in this module are computed
   using existing implementations from the
   Brain Connectivity Toolbox
   (`bctpy <https://github.com/aestrivex/bctpy>`_) and
   `NetworkX <https://networkx.org/>`_.

Classes
-------

.. currentmodule:: pyAvalanches

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   connectivity_metrics.MetricsObj
   connectivity_metrics.TrialMetrics
   connectivity_metrics.NetworkMetricsPipeline

.. _subject_group_results_ref:

:mod:`pyAvalanches.subject_group_results` -- Subject and Group-Level Results Containers
=================================================================

.. automodule:: pyAvalanches.subject_group_results
   :no-members:
   :no-inherited-members:

Aggregate results across subjects
and experimental conditions.


Classes
-------

.. currentmodule:: pyAvalanches

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   subject_group_results.Subject
   subject_group_results.GroupResults

.. _output_ref:

:mod:`pyAvalanches.output` -- Visualization & Reporting
=================================================

.. automodule:: pyAvalanches.output
   :no-members:
   :no-inherited-members:

Generate and export dataframes, figures and MNE Reports avalanche analysis results from subject or group-level data.


Functions
---------

.. currentmodule:: pyAvalanches

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   output.generate_report
   output.compute_summary_dataframes
   output.plot_group_size_loglog
   output.plot_group_duration_loglog
   output.plot_per_subject_size_kde
   output.plot_per_subject_duration_kde
   output.plot_group_size_kde
   output.plot_group_duration_kde
   output.plot_per_subject_atm_heatmap
   output.plot_group_atm_heatmap
   output.plot_subject_size_loglog
   output.plot_subject_duration_loglog
   output.plot_subject_atm_heatmap
   output.plot_subject_size_kde
   output.plot_subject_duration_kde