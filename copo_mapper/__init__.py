"""CO-PO mapping package."""

from .attainment import run_attainment_analysis, run_attainment_analysis_from_objects
from .pipeline import run_pairwise_mapping

__all__ = [
    "run_pairwise_mapping",
    "run_attainment_analysis",
    "run_attainment_analysis_from_objects",
]
