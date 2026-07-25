"""CO-PO mapping package."""

from .aggregate import (
    compute_program_po,
    compute_semester_po,
    run_program_aggregation,
    run_semester_aggregation,
)
from .attainment import run_attainment_analysis, run_attainment_analysis_from_objects
from .pipeline import run_pairwise_mapping

__all__ = [
    "run_pairwise_mapping",
    "run_attainment_analysis",
    "run_attainment_analysis_from_objects",
    "compute_semester_po",
    "compute_program_po",
    "run_semester_aggregation",
    "run_program_aggregation",
]
