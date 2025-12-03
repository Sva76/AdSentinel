import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import compute_sequence_features
from .model import AdSentinelRegressor


PROPERTIES = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "Titer"]
FOLD_COL   = "hierarchical_cluster_IgG_isotype_stratified_fold"


def run_cv(train_csv: str, out_csv: str):
    df = pd.read_csv(train_csv)

    # assume che le 5 proprietà siano già presenti nel CSV
    feats = compute_sequence_features(df).select_dtypes(include=[float, int]).copy()
    X_all = feats.values

    folds = sorted([f for f in df[FOLD_COL].dropna().unique()])

    cv_out = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence", FOLD_COL]].copy()

    for prop in PROPERTIES:
        if prop not in df.columns:
            print(f"[WARN] property {prop} not found, skipping.")
            continue

        y_all = df[prop].to_numpy(dtype=float)
        valid_mask = ~np.isnan(y_all)
        y_valid = y_all[valid_mask]
        X_valid = X_all[valid_mask]
        folds_valid = df.loc[valid_mask, FOLD_COL].to_numpy()

        oof_pred = np.full_like(y_all, np.nan, dtype=float)
        fold_rhos = []

        for f in folds:
            val_idx = (folds_valid == f)
            if not np.any(val_idx):
                continue
            train_idx = ~val_idx

            X_tr = X_valid[train_idx]
            y_tr = y_valid[train_idx]
            X_val = X_valid[val_idx]
            y_val = y_valid[val_idx]

            model = AdSentinelRegressor()
            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)

            global_val_idx = np.where(valid_mask)[0][val_idx]
            oof_pred[global_val_idx] = y_val_pred

            rho = spearmanr(y_val, y_val_pred).statistic
            fold_rhos.append(rho)
            print(f"{prop} – fold {f}: Spearman ρ = {rho:.3f}")

        mask_metric = ~np.isnan(oof_pred) & ~np.isnan(y_all)
        overall = spearmanr(y_all[mask_metric], oof_pred[mask_metric]).statistic
        print(f"{prop} – CV mean Spearman ρ = {overall:.3f} (folds={fold_rhos})")

        cv_out[prop] = oof_pred

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cv_out.to_csv(out_csv, index=False)
    print(f"Saved CV OOF predictions to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, help="GDPa1 train CSV with folds and targets")
    parser.add_argument("--out-csv", required=True, help="Output CSV with CV predictions")
    args = parser.parse_args()

    run_cv(args.train_csv, args.out_csv)


if __name__ == "__main__":
    main()
