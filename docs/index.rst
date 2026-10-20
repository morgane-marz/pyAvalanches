.. pyAvalanches documentation master file, created by
   sphinx-quickstart on Tue Jun 16 14:33:08 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/avalanche_logo.png
   :alt: pyAvalanches logo
   :width: 400px
   :align: center


Welcome to pyAvalanches
========================

**pyAvalanches** is a Python library for detecting, analyzing, and
characterizing **neuronal avalanches**, cascades of activity that
propagate through neural networks and exhibit scale-free dynamics.

.. note::

   This project is under active development.

Key Features
------------

- **Avalanche Detection & Characterization**: Identify neuronal cascades from EEG, MEG, ECoG, and other neural recordings. Study avalanche sizes, durations, patterns and power-law distributions with flexible thresholding and binning strategies
- **Spatiotemporal Analysis**: Map sensor-to-sensor propagation patterns and extract connectivity features via **Avalanche Transition Matrices**
- **Automated Reporting**: Generate publication-ready HTML reports for subject and group-level analyses and CSV summaries for easy data sharing and statistical follow-up
- **ML-Ready**: Scikit-learn compatible transformers for seamless integration into classification and analysis pipelines

Citation
-----------

If you use ``pyAvalanches`` in your research, please cite the accompanying
paper (preprint):

   Marzulli, M., Angiolelli, M., Mannino, C., Demuru, M., Sorrentino, P., &
   Corsi, M. C. (2026). pyAvalanches: A Python Package for Analyzing
   Spatiotemporal Propagation in Neuronal Avalanches. arXiv preprint
   arXiv:2609.11530.

.. code-block:: bibtex

   @article{marzulli2026pyavalanches,
     title={pyAvalanches: A Python Package for Analyzing Spatiotemporal Propagation in Neuronal Avalanches},
     author={Marzulli, M and Angiolelli, M and Mannino, C and Demuru, M and Sorrentino, P and Corsi, M-C},
     journal={arXiv preprint arXiv:2609.11530},
     year={2026}
   }

.. note::

   The final paper is currently under review. This entry will be updated with
   the full reference and DOI upon publication.

   Marzulli, M., Angiolelli, M., Mannino, C., Demuru, M., Sorrentino, P. &
   Corsi, M.C. *pyAvalanches: A Python Package for Analyzing Spatiotemporal
   Propagation in Neuronal Avalanches*. **Manuscript under review** (2026).

.. code-block:: bibtex

   @unpublished{marzulli2025pyavalanches,
     title  = {pyAvalanches: A Python Package for Analyzing Spatiotemporal
               Propagation in Neuronal Avalanches},
     author = {Marzulli, M and Angiolelli, M and Mannino, C and Demuru, M and Sorrentino, P and Corsi, M-C},
     year   = {2026},
     note   = {Manuscript submitted for publication}
   }

Quick Start
------------

Install the library:

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ pyAvalanches

And start the analysis!

.. code-block:: python

   import numpy as np
   from pyAvalanches.avalanches import AvalanchePipeline
   from pyAvalanches.transition_matrices import ATMPipeline
   from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline
   from pyAvalanches.subject_group_results import Subject

   # 1. Prepare your data
   input_data = {
       "data_array": your_data,           # 2D or 3D array
       "samp_freq": 256.0,               # sampling frequency (Hz)
       "fmin": 1.0, "fmax": 40.0,        # filter range (Hz)
       "channel_names": ["Fp1", "Fp2", "C3", "C4"],
       "trial_labels": ["left", "right", "left"],
   }
   # Or use MNE objects directly: input_data = mne.read_epochs("data.fif")

   # 2. Detect avalanches
   aval_pipeline = AvalanchePipeline(threshold=1.5, min_duration=2)
   avalanche_results = aval_pipeline.process(input_data)

   # 3. Compute transition matrices
   atm_pipeline = ATMPipeline(method='standard')
   atm_results = atm_pipeline.process(avalanche_results)

   # 4. Extract network metrics
   metrics_pipeline = NetworkMetricsPipeline(metrics=['clustering_coeff', 'node_strength'])
   metrics_results = metrics_pipeline.process(atm_results)

   # 5. Save results
   subject = Subject(
       subject_id="sub_01",
       avalanche_results=avalanche_results,
       atm_results=atm_results,
       metrics_results=metrics_results,
   )
   subject.save(out_path='./results')

.. tip::

   Process multiple subjects and perform group-level statistical comparisons using
   :class:`~pyAvalanches.subject_group_results.GroupResults`. See the :doc:`usage`
   guide for the complete step-by-step guide.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   concepts
   usage
   modules
   auto_examples/index