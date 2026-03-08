AdSentinel 2.7 – Cluster-aware cross-validation with Click mechanism.

The Click mechanism automatically selects the best feature configuration
per target property using nested cross-validation:
  - Outer loop: cluster-aware folds (honest evaluation)
  - Inner loop: selects optimal body/pinze/ESM combination per target
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import build_global_features, build_zoom_features, build_esm_features
from .model import ClickSelector


PROPERTIES = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "Titer"]
FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"


def _safe_spearman(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def run_cv(train_csv, out_csv, seed=42):
    np.random.seed(seed)

    df = pd.read_csv(train_csv)

    for col in ["antibody_name", "vh_protein_sequence", "vl_protein_sequence", FOLD_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Build feature blocks ---
    print("Building feature blocks...")
    X_global = build_global_features(df)
    X_zoom = build_zoom_features(df)
    X_esm = build_esm_features(df)

    has_esm = X_esm is not None
    if not has_esm:
        X_esm = np.zeros((len(df), 1))
        print("[WARN] No ESM embeddings found. Click will use pinze_only config.")

    print(f"  Global: {X_global.shape[1]} | Zoom: {X_zoom.shape[1]} | ESM: {X_esm.shape[1]}")

    folds = df[FOLD_COL].to_numpy(dtype=float)

    # --- Output frame ---
    cv_out = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence", FOLD_COL]].copy()

    click = ClickSelector(seed=seed)

    print("\n" + "=" * 60)
    print("AdSentinel Click – Adaptive feature selection")
    print("=" * 60)

    for prop in PROPERTIES:
        if prop not in df.columns:
            print(f"\n[WARN] property '{prop}' not found, skipping.")
            continue

        y = df[prop].to_numpy(dtype=float)

        if (~np.isnan(y)).sum() < 3:
            print(f"\n[WARN] property '{prop}': too few non-NaN labels, skipping.")
            continue

        print(f"\n--- {prop} ---")

        # Run Click: nested CV with automatic config selection
        oof_preds = click.run_cv(X_global, X_zoom, X_esm, y, folds, target_name=prop)

        # Per-fold Spearman
        unique_folds = sorted(set(folds[~np.isnan(folds)].astype(int)))
        for f in unique_folds:
            f_mask = (folds == f) & ~np.isnan(y) & ~np.isnan(oof_preds)
            if f_mask.sum() > 2:
                rho = _safe_spearman(y[f_mask], oof_preds[f_mask])
                print(f"  fold {f}: Spearman rho = {rho:.3f}")

        # Overall OOF Spearman
        mask = ~np.isnan(y) & ~np.isnan(oof_preds)
        overall = _safe_spearman(y[mask], oof_preds[mask])
        pval = spearmanr(y[mask], oof_preds[mask]).pvalue if mask.sum() > 2 else float("nan")
        print(f"  >>> OOF Spearman rho = {overall:.4f} (p={pval:.2e}, n={mask.sum()})")

        cv_out[prop] = oof_preds

    # --- Save ---
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    cv_out.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print(f"Saved Click CV predictions to: {out_csv}")

    # --- Summary ---
    print("\nClick configuration summary:")
    for prop, info in click.selected_configs.items():
        print(f"  {prop:20s}  config='{info['config']}'  votes={info['votes']}")


def main():
    parser = argparse.ArgumentParser(description="AdSentinel 2.7 - Click CV on GDPa1")
    parser.add_argument("--train-csv", required=True, help="GDPa1 train CSV with folds + targets")
    parser.add_argument("--out-csv", required=True, help="Output CSV with OOF predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_cv(train_csv=args.train_csv, out_csv=args.out_csv, seed=args.seed)


if __name__ == "__main__":
    main()
