"""flipbudget -- benchmark accuracy under an audited, imperfect scorer.

`flipbudget.pv` is the estimator for a scorer-output-stratified human audit
(the design used in this project): A* = a(1-q) + (1-a)r. The sensitivity/
specificity functions below (`g`, ...) are correct only when fed true error
rates alpha, beta -- NOT stratum-conditional audit rates (CORE_ESTIMAND_AUDIT.md).

The headline function is `corrected_interval`: given someone's benchmark
records and measured scorer-error margins, it returns the identification-
corrected interval for a model's accuracy or for a pairwise comparison --
not just a report of the margins themselves. Correcting someone else's
number is the adoption path; emitting a descriptive card is secondary.
"""
from .identification import g, single_model_extrema, comparison_extrema, flip_budget
from . import pv
from .inference import corrected_interval, paired_bootstrap_delta

__all__ = [
    "pv",
    "g",
    "single_model_extrema",
    "comparison_extrema",
    "flip_budget",
    "corrected_interval",
    "paired_bootstrap_delta",
]

__version__ = "0.1.0"
