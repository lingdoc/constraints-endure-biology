"""
Package initialization exposing the core PyMC mixed model regression engine functions.
"""
from .bayesian_engine import fit_trivariance_pglmm_gpu, summarize_trivariance_metrics

__all__ = ["fit_trivariance_pglmm_gpu", "summarize_trivariance_metrics"]
