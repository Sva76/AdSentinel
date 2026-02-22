        import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import compute_sequence_features
from .model import AdSentinelRegressor


PROPERTIES = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "Titer"]
FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman robusto: ritorna nan se non calcolabile."""
    if len(y_true) < 2:
        return float("nan")
    # se vettore costante, spearman può essere nan
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def run_cv(train_csv: str, out_csv: str, seed: int = 42):
    np.random.seed(seed)

    df = pd.read_csv(train_csv)

    # Controlli minimi
    for col in ["antibody_name", "vh_protein_sequence", "vl_protein_sequence", FOLD_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Features (DataFrame numerico)
    feats = compute_sequence_features(df)
    X_all = feats.values

    # folds presenti
    folds = sorted([f for f in df[FOLD_COL].dropna().unique().tolist()])
    if len(folds) == 0:
        raise ValueError(f"No folds found in column {FOLD_COL}")

    # Output: OOF predictions per property
    cv_out = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence", FOLD_COL]].copy()

    for prop in PROPERTIES:
        if prop not in df.columns:
            print(f"[WARN] property '{prop}' not found, skipping.")
            continue

        y_all = df[prop].to_numpy(dtype=float)
        valid_mask = ~np.isnan(y_all)

        if valid_mask.sum() < 3:
            print(f"[WARN] property '{prop}': too few non-NaN labels, skipping.")
            continue

        X_valid = X_all[valid_mask]
        y_valid = y_all[valid_mask]
        folds_valid = df.loc[valid_mask, FOLD_COL].to_numpy()

        oof_pred = np.full(shape=y_all.shape, fill_value=np.nan, dtype=float)
        fold_rhos = []

        for f in folds:
            val_mask = (folds_valid == f)
            if not np.any(val_mask):
                continue
            tr_mask = ~val_mask

            X_tr, y_tr = X_valid[tr_mask], y_valid[tr_mask]
            X_val, y_val = X_valid[val_mask], y_valid[val_mask]

            model = AdSentinelRegressor()
            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)

            # riportiamo le predizioni sul vettore originale df
            global_val_idx = np.where(valid_mask)[0][val_mask]
            oof_pred[global_val_idx] = y_val_pred

            rho = _safe_spearman(y_val, y_val_pred)
            fold_rhos.append(rho)
            print(f"{prop} – fold {f}: Spearman ρ = {rho:.3f}")

        # Overall CV (OOF)
        mask_metric = ~np.isnan(oof_pred) & ~np.isnan(y_all)
        overall = _safe_spearman(y_all[mask_metric], oof_pred[mask_metric])
        print(f"{prop} – OOF Spearman ρ = {overall:.3f} | folds={fold_rhos}")

        cv_out[prop] = oof_pred

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cv_out.to_csv(out_csv, index=False)
    print(f"Saved CV OOF predictions to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="AdSentinel 2.6 - 5-fold CV on GDPa1")
    parser.add_argument("--train-csv", required=True, help="GDPa1 train CSV with folds + targets")
    parser.add_argument("--out-csv", required=True, help="Output CSV with OOF predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_cv(train_csv=args.train_csv, out_csv=args.out_csv, seed=args.seed)


if __name__ == "__main__":
    main()
