AdSentinel 2.7 – Click-adaptive stacked regressor.

Two modes:
  1. Standard: Ridge → XGBoost with OOF stacking (backward compatible)
  2. Click: nested CV selects the best feature configuration per target

Feature configurations:
  - full_esm:      global + full ESM-2 (1289 features)
  - body20_pinze:  global + CDR zoom + ESM PCA-20 (71 features)
  - body50_pinze:  global + CDR zoom + ESM PCA-50 (101 features)
  - pinze_only:    global + CDR zoom, no ESM (51 features)
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from xgboost import XGBRegressor


# --- Click configurations ---

CLICK_CONFIGS = {
    "full_esm":      {"pca": None,  "zoom": False},
    "body20_pinze":  {"pca": 20,    "zoom": True},
    "body50_pinze":  {"pca": 50,    "zoom": True},
    "pinze_only":    {"pca": 0,     "zoom": True},
}


def assemble_features(X_global, X_zoom, X_esm, config_name):
    """Build feature matrix for a given Click configuration."""
    cfg = CLICK_CONFIGS[config_name]
    parts = [X_global]

    if cfg["zoom"]:
        parts.append(X_zoom)

    pca_n = cfg["pca"]
    if pca_n is None:
        parts.append(X_esm)
    elif pca_n > 0:
        n = min(pca_n, X_esm.shape[1], X_esm.shape[0] - 1)
        parts.append(PCA(n_components=n, random_state=42).fit_transform(X_esm))
    # pca_n == 0: no ESM

    return np.hstack(parts)


# --- Base regressor (unchanged OOF stacking) ---

class AdSentinelRegressor:

    def __init__(self, n_splits=5, ridge_alpha=1.0, seed=42):
        self.n_splits = n_splits
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.ridge = Ridge(alpha=ridge_alpha)
        self.xgb = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
        )

    def fit(self, X, y):
        oof_ridge = np.zeros(len(y))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in kf.split(X):
            ridge_fold = Ridge(alpha=self.ridge_alpha)
            ridge_fold.fit(X[train_idx], y[train_idx])
            oof_ridge[val_idx] = ridge_fold.predict(X[val_idx])

        self.ridge.fit(X, y)

        X_stacked = np.hstack([X, oof_ridge.reshape(-1, 1)])
        self.xgb.fit(X_stacked, y)

    def predict(self, X):
        ridge_pred = self.ridge.predict(X).reshape(-1, 1)
        return self.xgb.predict(np.hstack([X, ridge_pred]))


# --- Click mechanism ---

class ClickSelector:
    """
    Nested CV to find the best feature configuration per target.

    Outer loop: cluster-aware folds (honest evaluation)
    Inner loop: leave-one-fold-out from training folds (config selection)
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.selected_configs = {}

    def select_config(self, X_global, X_zoom, X_esm, y, folds, target_name=""):
        """
        Run inner CV to find the best config for one target.
        Returns the config name selected by majority vote across outer folds.
        """
        unique_folds = sorted(set(folds[~np.isnan(folds)].astype(int)))
        votes = {}

        for outer_fold in unique_folds:
            outer_train = folds != outer_fold
            inner_folds = sorted(set(folds[outer_train & ~np.isnan(folds)].astype(int)))

            scores = {}
            for config_name in CLICK_CONFIGS:
                try:
                    X_cfg = assemble_features(X_global, X_zoom, X_esm, config_name)
                except Exception:
                    continue

                rhos = []
                for inner_fold in inner_folds:
                    itr = outer_train & (folds != inner_fold) & ~np.isnan(y)
                    iva = outer_train & (folds == inner_fold) & ~np.isnan(y)

                    if itr.sum() < 10 or iva.sum() < 3:
                        continue

                    r = Ridge(alpha=1.0)
                    r.fit(X_cfg[itr], y[itr])
                    rho, _ = spearmanr(y[iva], r.predict(X_cfg[iva]))
                    if not np.isnan(rho):
                        rhos.append(rho)

                if rhos:
                    scores[config_name] = np.mean(rhos)

            best = max(scores, key=scores.get) if scores else "body20_pinze"
            votes[best] = votes.get(best, 0) + 1

        # Majority vote
        winner = max(votes, key=votes.get)
        self.selected_configs[target_name] = {
            "config": winner,
            "votes": votes,
        }

        print(f"  [CLICK] {target_name}: selected '{winner}' | votes: {votes}")
        return winner

    def run_cv(self, X_global, X_zoom, X_esm, y, folds, target_name=""):
        """
        Full Click CV: select config via inner loop, evaluate via outer loop.
        Returns out-of-fold predictions.
        """
        unique_folds = sorted(set(folds[~np.isnan(folds)].astype(int)))
        oof_preds = np.full(len(y), np.nan)
        votes = {}

        for outer_fold in unique_folds:
            outer_val = folds == outer_fold
            outer_train = folds != outer_fold
            inner_folds = sorted(set(folds[outer_train & ~np.isnan(folds)].astype(int)))

            # --- Inner: select best config ---
            scores = {}
            for config_name in CLICK_CONFIGS:
                try:
                    X_cfg = assemble_features(X_global, X_zoom, X_esm, config_name)
                except Exception:
                    continue

                rhos = []
                for inner_fold in inner_folds:
                    itr = outer_train & (folds != inner_fold) & ~np.isnan(y)
                    iva = outer_train & (folds == inner_fold) & ~np.isnan(y)
                    if itr.sum() < 10 or iva.sum() < 3:
                        continue
                    r = Ridge(alpha=1.0)
                    r.fit(X_cfg[itr], y[itr])
                    rho, _ = spearmanr(y[iva], r.predict(X_cfg[iva]))
                    if not np.isnan(rho):
                        rhos.append(rho)
                if rhos:
                    scores[config_name] = np.mean(rhos)

            best = max(scores, key=scores.get) if scores else "body20_pinze"
            votes[best] = votes.get(best, 0) + 1

            # --- Outer: train and predict with best config ---
            X_cfg = assemble_features(X_global, X_zoom, X_esm, best)
            tr_valid = outer_train & ~np.isnan(y)

            model = AdSentinelRegressor(seed=self.seed)
            model.fit(X_cfg[tr_valid], y[tr_valid])
            oof_preds[outer_val] = model.predict(X_cfg[outer_val])

        self.selected_configs[target_name] = {
            "config": max(votes, key=votes.get),
            "votes": votes,
        }

        print(f"  [CLICK] {target_name}: {votes}")
        return oof_preds
