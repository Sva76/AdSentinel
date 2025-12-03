import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import compute_sequence_features
from .model import AdSentinelRegressor

PROPERTIES = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "Titer"]


def run_train_and_predict(train_csv: str, heldout_csv: str, out_train_csv: str, out_heldout_csv: str):
    train_df   = pd.read_csv(train_csv)
    heldout_df = pd.read_csv(heldout_csv)

    feats_train = compute_sequence_features(train_df).select_dtypes(include=[float, int])
    feats_held  = compute_sequence_features(heldout_df).select_dtypes(include=[float, int])

    # align columns
    feats_held = feats_held[feats_train.columns]

    X_train = feats_train.values
    X_held  = feats_held.values

    cv_out = train_df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
    test_out = heldout_df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()

    for prop in PROPERTIES:
        if prop not in train_df.columns:
            print(f"[WARN] property {prop} not found, skipping.")
            continue

        y_all = train_df[prop].to_numpy(dtype=float)
        valid_mask = ~np.isnan(y_all)
        X_valid = X_train[valid_mask]
        y_valid = y_all[valid_mask]

        model = AdSentinelRegressor()
        model.fit(X_valid, y_valid)

        # store back train preds (on all samples)
        y_train_pred = np.full_like(y_all, np.nan, dtype=float)
        y_train_pred[valid_mask] = model.predict(X_valid)
        cv_out[prop] = y_train_pred

        # heldout predictions
        y_held_pred = model.predict(X_held)
        test_out[prop] = y_held_pred

    Path(out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_heldout_csv).parent.mkdir(parents=True, exist_ok=True)

    cv_out.to_csv(out_train_csv, index=False)
    test_out.to_csv(out_heldout_csv, index=False)
    print(f"Saved train preds to {out_train_csv}")
    print(f"Saved heldout preds to {out_heldout_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv",   required=True)
    parser.add_argument("--heldout-csv", required=True)
    parser.add_argument("--out-train-csv",   required=True)
    parser.add_argument("--out-heldout-csv", required=True)
    args = parser.parse_args()

    run_train_and_predict(
        train_csv=args.train_csv,
        heldout_csv=args.heldout_csv,
        out_train_csv=args.out_train_csv,
        out_heldout_csv=args.out_heldout_csv,
    )


if __name__ == "__main__":
    main()
