AdSentinel 2.7 – Train on full dataset and predict heldout.

Uses the Click mechanism to select the best feature configuration
per target property before training the final model.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_global_features, build_zoom_features, build_esm_features
from .model import AdSentinelRegressor, ClickSelector, assemble_features


PROPERTIES = ["HIC", "AC-SINS_pH7.4", "PR_CHO", "Tm2", "Titer"]
FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"


def run_train_and_predict(
    train_csv,
    heldout_csv,
    out_train_csv,
    out_heldout_csv,
    seed=42,
):
    np.random.seed(seed)

    train_df = pd.read_csv(train_csv)
    heldout_df = pd.read_csv(heldout_csv)

    for col in ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing required column: {col}")
        if col not in heldout_df.columns:
            raise ValueError(f"Heldout CSV missing required column: {col}")

    # --- Build feature blocks for train ---
    print("Building train feature blocks...")
    X_global_train = build_global_features(train_df)
    X_zoom_train = build_zoom_features(train_df)
    X_esm_train = build_esm_features(train_df)

    has_esm = X_esm_train is not None
    if not has_esm:
        X_esm_train = np.zeros((len(train_df), 1))
        print("[WARN] No ESM embeddings for train set.")

    print(f"  Train - Global: {X_global_train.shape[1]} | Zoom: {X_zoom_train.shape[1]} | ESM: {X_esm_train.shape[1]}")

    # --- Build feature blocks for heldout ---
    print("Building heldout feature blocks...")
    X_global_held = build_global_features(heldout_df)
    X_zoom_held = build_zoom_features(heldout_df)
    X_esm_held = build_esm_features(heldout_df)

    if not has_esm or X_esm_held is None:
        X_esm_held = np.zeros((len(heldout_df), 1))
        print("[WARN] No ESM embeddings for heldout set.")

    print(f"  Heldout - Global: {X_global_held.shape[1]} | Zoom: {X_zoom_held.shape[1]} | ESM: {X_esm_held.shape[1]}")

    # --- Click: select best config per target ---
    folds = train_df[FOLD_COL].to_numpy(dtype=float) if FOLD_COL in train_df.columns else None

    click = ClickSelector(seed=seed)

    # Output frames
    train_out = train_df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
    heldout_out = heldout_df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()

    print("\n" + "=" * 60)
    print("AdSentinel Click – Training + Heldout Prediction")
    print("=" * 60)

    for prop in PROPERTIES:
        if prop not in train_df.columns:
            print(f"\n[WARN] property '{prop}' not found in train set, skipping.")
            continue

        y_all = train_df[prop].to_numpy(dtype=float)
        valid_mask = ~np.isnan(y_all)

        if valid_mask.sum() < 3:
            print(f"\n[WARN] property '{prop}': too few non-NaN labels, skipping.")
            continue

        print(f"\n--- {prop} ---")

        # Click selection (uses inner CV on training data)
        if folds is not None:
            best_config = click.select_config(
                X_global_train, X_zoom_train, X_esm_train,
                y_all, folds, target_name=prop,
            )
        else:
            best_config = "full_esm" if has_esm else "pinze_only"
            print(f"  [CLICK] No fold column, using default: {best_config}")

        # Assemble features with selected config
        X_train = assemble_features(X_global_train, X_zoom_train, X_esm_train, best_config)
        X_held = assemble_features(X_global_held, X_zoom_held, X_esm_held, best_config)

        print(f"  Config '{best_config}' -> {X_train.shape[1]} features")

        # Train on valid samples
        X_valid = X_train[valid_mask]
        y_valid = y_all[valid_mask]

        model = AdSentinelRegressor(seed=seed)
        model.fit(X_valid, y_valid)

        # Train predictions (only where y exists)
        y_train_pred = np.full_like(y_all, np.nan, dtype=float)
        y_train_pred[valid_mask] = model.predict(X_valid)
        train_out[prop] = y_train_pred

        # Heldout predictions
        heldout_out[prop] = model.predict(X_held)

        print(f"  [OK] Trained + predicted for: {prop}")

    # --- Save ---
    Path(out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_heldout_csv).parent.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(out_train_csv, index=False)
    heldout_out.to_csv(out_heldout_csv, index=False)

    print(f"\nSaved train predictions to:  {out_train_csv}")
    print(f"Saved heldout predictions to: {out_heldout_csv}")

    # Summary
    print("\nClick configuration summary:")
    for prop, info in click.selected_configs.items():
        print(f"  {prop:20s}  config='{info['config']}'  votes={info['votes']}")


def main():
    parser = argparse.ArgumentParser(description="AdSentinel 2.7 - Click train + heldout predict")
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
