AdSentinel 2.7 – Adaptive antibody developability prediction with Click mechanism.

This package provides:
- Three-block feature extraction (global, CDR zoom, ESM embeddings)
- Ridge + XGBoost regressor with OOF stacking
- Click mechanism for adaptive feature selection per target
- Cluster-aware cross-validation and heldout prediction
"""

__version__ = "2.7.0"

from .features import compute_sequence_features, build_global_features, build_zoom_features, build_esm_features
from .model import AdSentinelRegressor, ClickSelector

__all__ = [
    "compute_sequence_features",
    "build_global_features",
    "build_zoom_features",
    "build_esm_features",
    "AdSentinelRegressor",
    "ClickSelector",
]
