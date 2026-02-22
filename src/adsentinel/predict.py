import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import compute_sequence_features
from .model import AdSentinelRegressor


PROPERTIES = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "Titer"]


def run_train_and_predict(
    train_csv: str,
    heldout_csv: str,
    out_train_csv: str,
    out_heldout_csv: str,
    seed: int = 42,
):
    np.random.seed(seed)

    train_df = pd.read_csv(train_csv)
    heldout_df = pd.read_csv(heldout_csv)

    for col in ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing required column: {col}")
        if col not in heldout_df.columns:
            raise ValueError(f"Heldout CSV missing required column: {col}")

    # Features
    feats_train = compute_sequence_features(train_df)
    feats_held = compute_sequence_features(heldout_df)

    # Allineamento colonne (difensivo)
    if feats_train.shape[1] != feats_held.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: train={feats_train.shape[1]} heldout={feats_held.shape[1]}"
        )

    X_train = feats_train.values
    X_held = feats_held.values

    # Output frames
    train_out = train_df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
    heldout_out = heldout_df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()

    for prop in PROPERTIES:
        if prop not in train_df.columns:
            print(f"[WARN] property '{prop}' not found in train set, skipping.")
            continue

        y_all = train_df[prop].to_numpy(dtype=float)
        valid_mask = ~np.isnan(y_all)

        if valid_mask.sum() < 3:
            print(f"[WARN] property '{prop}': too few non-NaN labels, skipping.")
            continue

        X_valid = X_train[valid_mask]
        y_valid = y_all[valid_mask]

        model = AdSentinelRegressor()
        model.fit(X_valid, y_valid)

        # Train predictions (solo dove y esiste; altrove NaN)
        y_train_pred = np.full_like(y_all, np.nan, dtype=float)
        y_train_pred[valid_mask] = model.predict(X_valid)
        train_out[prop] = y_train_pred

        # Heldout predictions (sempre)
        heldout_out[prop] = model.predict(X_held)

        print(f"[OK] Trained + predicted for: {prop}")

    Path(out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_heldout_csv).parent.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(out_train_csv, index=False)
    heldout_out.to_csv(out_heldout_csv, index=False)

    print(f"Saved train predictions to:  {out_train_csv}")
    print(f"Saved heldout predictions to: {out_heldout_csv}")


def main():
    parser = argparse.ArgumentParser(description="AdSentinel 2.6 - train on full train set and predict heldout")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--heldout-csv", required=True)
    parser.add_argument("--out-train-csv", required=True)
    parser.add_argument("--out-heldout-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_train_and_predict(
        train_csv=args.train_csv,
        heldout_csv=args.heldout_csv,
        out_train_csv=args.out_train_csv,
        out_heldout_csv=args.out_heldout_csv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
