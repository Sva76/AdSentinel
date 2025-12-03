from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:  # xgboost non obbligatorio
    XGBRegressor = None
    _HAS_XGB = False


@dataclass
class AdSentinelRegressor:
    """
    Hybrid regressor: RidgeCV (global trend) + optional XGBoost (local corrections).

    This is a compact version of the model you used in Colab:
    - imputazione mediana
    - standardizzazione
    - PCA
    - RidgeCV
    - XGBoost (se disponibile)
    """
    n_components: int = 32
    ridge_alphas: Iterable[float] = tuple(10.0 ** p for p in range(-3, 4))
    use_xgb: bool = True

    # learned objects
    imputer_: Optional[SimpleImputer] = None
    scaler_: Optional[StandardScaler] = None
    pca_: Optional[PCA] = None
    ridge_: Optional[RidgeCV] = None
    xgb_: Optional[XGBRegressor] = None

    def _fit_preproc(self, X: np.ndarray) -> np.ndarray:
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        n_comp = min(self.n_components, X.shape[1])
        self.pca_ = PCA(n_components=n_comp)

        X_imp = self.imputer_.fit_transform(X)
        X_sc  = self.scaler_.fit_transform(X_imp)
        X_pca = self.pca_.fit_transform(X_sc)
        return X_pca

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer_.transform(X)
        X_sc  = self.scaler_.transform(X_imp)
        X_pca = self.pca_.transform(X_sc)
        return X_pca

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_pca = self._fit_preproc(X)

        self.ridge_ = RidgeCV(alphas=list(self.ridge_alphas), store_cv_values=False)
        self.ridge_.fit(X_pca, y)

        if self.use_xgb and _HAS_XGB:
            self.xgb_ = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
            )
            self.xgb_.fit(X_pca, y)
        else:
            self.xgb_ = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pca = self._transform(X)
        y_ridge = self.ridge_.predict(X_pca)

        if self.xgb_ is None:
            return y_ridge

        y_xgb = self.xgb_.predict(X_pca)
        # semplice media 50/50 come nel design originale
        return 0.5 * y_ridge + 0.5 * y_xgb
