"""
AdSentinel 2.6
Hybrid sequence-based model for antibody developability prediction.

This package provides:
- Feature extraction from VH/VL sequences
- A hybrid Ridge + XGBoost regressor
- Simple helpers for cross-validation and heldout prediction
"""

from .features import compute_sequence_features
from .model import AdSentinelRegressor

__all__ = ["compute_sequence_features", "AdSentinelRegressor"]
