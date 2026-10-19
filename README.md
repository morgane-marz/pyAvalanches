<!-- TODO: Optional - Add a logo here if you have one -->
<p align="center">
  <img src="images/avalanche_logo.png" width="300">
</p>

<p align="center">
  <!-- TODO: Add badges. Replace `your-username/pyAvalanches` with your actual repo path. -->
  <a href="https://test.pypi.org/project/pyAvalanches/"><img src="https://img.shields.io/pypi/v/pyAvalanches.svg?pypiBaseUrl=https://test.pypi.org&amp;label=TestPyPI" alt="TestPyPI"></a>
  <a href="https://github.com/your-username/pyAvalanches/actions"><img src="https://github.com/your-username/pyAvalanches/actions/workflows/ci.yml/badge.svg" alt="Build Status"></a>
  <a href="https://your-docs-url.readthedocs.io/"><img src="https://readthedocs.org/projects/your-docs-url/badge/?version=latest" alt="Documentation Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License"></a>
</p>

# pyAvalanches

**pyAvalanches** is a Python library for detecting, analyzing, and characterizing neuronal avalanches, cascades of activity that propagate through neural networks and exhibit scale-free dynamics.

---

### Project Status

This project is under active development. The API may change, and new features are being added regularly. Contributions and feedback are welcome!

## Key Features

*   **Avalanche Detection**
    *   Detect neuronal avalanches from continuous and epoched recordings.
    *   Support for EEG, MEG, ECoG, and other sensor-level data.
    *   Flexible thresholding (e.g., standard deviation, absolute value) and binning strategies.
    *   Compute avalanche size and duration distributions.
    *   Power-law fitting and visualization tools.

*   **Avalanche Transition Matrix (ATM)**
    *   Compute sensor-to-sensor transition probabilities from detected avalanches.
    *   Map spatiotemporal propagation patterns across channels.
    *   Extract connectivity features from avalanche dynamics.

*   **Machine Learning Integration**
    *   Scikit-learn compatible transformers and feature extractors.
    *   Use ATM features in your classification pipelines.
    *   Ready for cross-validation, grid search, and standard ML workflows.

## 🔧 Installation

You can install `pyAvalanches` directly from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ pyAvalanches```
```

## 🚀 Quick Start

Here's a simple example of how to detect avalanches and compute Avalanche Transition Matrices (ATM) from sample data.

```python
import numpy as np
from pyAvalanches.avalanches import AvalanchePipeline
from pyAvalanches.transition_matrices import ATMPipeline
from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline
from pyAvalanches.subject_group_results import Subject
from pyAvalanches.output import generate_report

# 1. Load or generate your data (e.g., 3 trials, 4 channels, 2 seconds at 256 Hz) and metadata

n_trials = 3
n_channels = 4
n_timepoints = 512

data = np.random.randn(n_trials, n_channels, n_timepoints)

input_dict = {
    "data_array": data,
    "samp_freq": 256.0,
    "fmin": 1.0,
    "fmax": 40.0,
    "channel_names": ["Fp1", "Fp2", "C3", "C4"],
    "trial_labels": ["left", "right", "left"],
    "t_max": None
}

# 2. Set up the avalanche pipeline
#    - Threshold data at 1.5 standard deviations
#    - Mininum duration of an avalanche is 2 time points

avalanche_pipeline = AvalanchePipeline(
    z_scoring="by_trial",
    threshold_method='std',
    threshold=1.5,
    min_duration=2)

# 3. Detect avalanches from the data

aval_results = avalanche_pipeline.process(input_dict)
print(f"Processed {len(aval_results.trial_indexes)} trials.")

# 4. Set up the Avalanche Transition Matrices pipeline
#    - Standard computation: each ATM element ij is the probability that region j activated 
#    following the activation of region i
#    - ATM get symmetrized

atm_pipeline = ATMPipeline(
    method='standard',
    symm=True,
)

# 5. Compute the Avalanche Transition Matrices (ATMs)

atm_results = atm_pipeline.process(aval_results)
print(f"Processed {len(atm_results.trial_indexes)} trials.")
print(f"Mean ATM has shape: {atm_results.global_mean_atm().shape}") # n_channels x n_channels

# 6. Compute connectivity metrics from the ATMs

metrics_pipeline = NetworkMetricsPipeline(metrics=["node_strength", "eigenvect_centrality"], symm=True)
metrics_results = metrics_pipeline.process(atm_results)
print(f"Processed {len(metrics_results.trial_indexes)} trials.")

# 7. Store your results at subject-level

subj_0 = Subject(
    subject_id=0,
    subject_label="patient",
    dataset=input_dict,
    avalanche_results=aval_results,
    atm_results=atm_results,
    metrics_results=metrics_results
)

# 8. Generate an MNE Report with an overview of all your results

report = generate_report(
    results=subj_0,
    out_path='./results',
    prefix='example_subject_results',
    overwrite=True,
)

print("Group report saved to ./results/group_openneuro_report.html")



```

## 📚 Documentation

For a detailed guide, tutorials, and a full API reference, please see our documentation at:

[**your-docs-url.readthedocs.io**](https://your-docs-url.readthedocs.io)

<!-- TODO: Create documentation (e.g., using Sphinx/Read the Docs) and replace the URL. -->

## 📜 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

<!-- TODO: Create a LICENSE file (e.g., by choosing a license on GitHub). -->


## 🎓 Citation

If you use `pyAvalanches` in your research, please cite the accompanying paper (preprint):

>Marzulli, M., Angiolelli, A., Mannino, C., Demuru, M., Sorrentino, P., & Corsi, M. C. (2026). pyAvalanches: A Python Package 
>for Analyzing Spatiotemporal Propagation in Neuronal Avalanches. arXiv preprint arXiv:2609.11530.


```bibtex
@article{marzulli2026pyavalanches,
  title={pyAvalanches: A Python Package for Analyzing Spatiotemporal Propagation in Neuronal Avalanches},
  author={Marzulli, M and Angiolelli, A and Mannino, C and Demuru, M and Sorrentino, P and Corsi, M-C},
  journal={arXiv preprint arXiv:2609.11530},
  year={2026}
}
```

Note: The final paper is currently under review. This entry will be updated with the
full reference and DOI upon publication.

> Marzulli, M., Angiolelli, M., Mannino, C., Demuru, M., Sorrentino, P. & Corsi, M.C. *pyAvalanches: A Python Package for Analyzing
> Spatiotemporal Propagation in Neuronal Avalanches*. **Manuscript under review** (2026).

```bibtex
@unpublished{marzulli2025pyavalanches,
  title  = {pyAvalanches: A Python Package for Analyzing Spatiotemporal
            Propagation in Neuronal Avalanches},
  author = {Marzulli, M and Angiolelli, A and Mannino, C and Demuru, M and Sorrentino, P and Corsi, M-C},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```