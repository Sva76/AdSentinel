# Data

This folder is **not** versioned with the GDPa1 data (because of size and license).

## Required files

Download from the Ginkgo / Hugging Face AbDev challenge space and place here:

- `GDPa1_v1.2_20250814.csv` — training set with targets and fold assignments
- `heldout-set-sequences.csv` — heldout test set (sequences only)
- `AbSentinel_vectors_1280.csv` — pre-computed ESM-2 mean-pooled embeddings (246 antibodies × 1280 dimensions)

## Without ESM embeddings

If you don't have the ESM vectors file, AdSentinel will run with global + CDR zoom features only (no ESM). The Click mechanism will automatically select `pinze_only` configuration.

## Generating ESM embeddings from scratch

If you want to regenerate the embeddings:

```bash
pip install torch fair-esm
```

Then run ESM-2 (esm2_t33_650M_UR50D) on concatenated VH+VL sequences, mean-pool the per-residue outputs, and save as CSV with columns: `label, d0, d1, ..., d1279` where `label` is the antibody name.

## Usage

```bash
# Click cross-validation
python -m adsentinel.train_cv \
  --train-csv data/GDPa1_v1.2_20250814.csv \
  --out-csv outputs/gdpa1_cv_predictions.csv

# Train + heldout prediction
python -m adsentinel.predict \
  --train-csv data/GDPa1_v1.2_20250814.csv \
  --heldout-csv data/heldout-set-sequences.csv \
  --out-train-csv outputs/gdpa1_train_preds.csv \
  --out-heldout-csv outputs/gdpa1_heldout_preds.csv
