"""CO-PO mapping package."""

from .attainment import run_attainment_analysis
from .pipeline import run_pairwise_mapping

__all__ = ["run_pairwise_mapping", "run_attainment_analysis"]
