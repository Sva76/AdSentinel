"""
AdSentinel 2.6 – Ridge → XGBoost stacked regressor.

Fix: Ridge predictions for XGBoost training are now generated
out-of-fold to prevent data leakage in the stacking step.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


class AdSentinelRegressor:

    def __init__(self, n_splits: int = 5, ridge_alpha: float = 1.0, seed: int = 42):
        self.n_splits = n_splits
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        self.ridge = Ridge(alpha=ridge_alpha)
        self.xgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,          # ridotto da 4 → 3 per dataset piccoli
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
        )

    def fit(self, X, y):
        # --- Step 1: genera ridge predictions out-of-fold ---
        oof_ridge = np.zeros(len(y))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in kf.split(X):
            ridge_fold = Ridge(alpha=self.ridge_alpha)
            ridge_fold.fit(X[train_idx], y[train_idx])
            oof_ridge[val_idx] = ridge_fold.predict(X[val_idx])

        # --- Step 2: fit Ridge finale su tutto (per inference) ---
        self.ridge.fit(X, y)

        # --- Step 3: fit XGBoost con predizioni OOF (no leakage) ---
        X_stacked = np.hstack([X, oof_ridge.reshape(-1, 1)])
        self.xgb.fit(X_stacked, y)

    def predict(self, X):
        ridge_pred = self.ridge.predict(X).reshape(-1, 1)
        return self.xgb.predict(np.hstack([X, ridge_pred]))
