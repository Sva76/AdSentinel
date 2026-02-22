import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

class AdSentinelRegressor:

    def __init__(self):
        self.ridge = Ridge(alpha=1.0)
        self.xgb = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    def fit(self, X, y):
        self.ridge.fit(X, y)
        ridge_pred = self.ridge.predict(X).reshape(-1, 1)
        self.xgb.fit(np.hstack([X, ridge_pred]), y)

    def predict(self, X):
        ridge_pred = self.ridge.predict(X).reshape(-1, 1)
        return self.xgb.predict(np.hstack([X, ridge_pred]))
