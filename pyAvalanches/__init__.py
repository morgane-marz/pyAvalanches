"""A Python library for detecting, analyzing, and characterizing neuronal avalanches."""

__version__ = "0.1.0"
__author__ = "Morgane Marzulli"
__email__ = "morgane.marzulli@icm-institute.org"

from . import dataset
from . import avalanches
from . import transition_matrices
from . import connectivity_metrics
from . import output
from . import subject_group_results

__all__ = [
    "avalanches",
    "connectivity_metrics",
    "dataset",
    "subject_group_results",
    "output",
    "transition_matrices",
]