import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

class AdSentinelModel:
    def __init__(self, n_components=64):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
        self.xgb = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8
        )

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        self.ridge.fit(Xp, y)
        self.xgb.fit(Xp, y)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        p1 = self.ridge.predict(Xp)
        p2 = self.xgb.predict(Xp)
        return 0.5 * p1 + 0.5 * p2
