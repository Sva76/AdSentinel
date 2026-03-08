# AdSentinel 2.7 – Antibody Developability Model with Click Mechanism

AdSentinel 2.7 is an adaptive sequence-based model for predicting antibody developability properties on the **GDPa1 dataset** (Ginkgo Antibody Developability Benchmark).

The model operates on three feature blocks:

- **Block 1 – Body** (global): sequence physicochemical features + isotype encoding
- **Block 2 – Pinze** (zoom): CDR-specific descriptors (hydropathy patches, charge density, sticky regions, loop lengths)
- **Block 3 – ESM**: ESM-2 mean-pooled embeddings (1280-dim, pre-computed)

The **Click mechanism** automatically selects the optimal combination of these blocks for each target property using nested cross-validation — because different biophysical properties depend on different levels of molecular detail.

---

# How Click Works

Each antibody property responds to a different level of structural information:

- **HIC** (hydrophobicity) is driven by global sequence composition → Click selects full ESM
- **Tm2** (thermostability) depends on 3D packing → Click selects compressed ESM + CDR zoom
- **AC-SINS** (self-association) depends on surface patches → Click adapts per fold

The mechanism uses nested CV:

- **Outer loop**: cluster-aware folds (honest evaluation)
- **Inner loop**: tests each feature configuration and selects the best per target

Available configurations:

| Config | Features | Dimensions |
|--------|----------|------------|
| `full_esm` | Global + full ESM-2 | ~1289 |
| `body50_pinze` | Global + CDR zoom + ESM PCA-50 | ~101 |
| `body20_pinze` | Global + CDR zoom + ESM PCA-20 | ~71 |
| `pinze_only` | Global + CDR zoom, no ESM | ~51 |

---

# Target Properties (GDPa1)

The model predicts five biophysical properties:

- **HIC** – Hydrophobic Interaction Chromatography retention time
- **AC-SINS_pH7.4** – Self-association
- **PR_CHO** – Polyreactivity in CHO cells
- **Tm2** – CH2 domain thermostability
- **Titer** – Expression yield

---

# Quickstart

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Sva76/AdSentinel.git
cd AdSentinel
```

## 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run Click cross-validation (GDPa1)

Place the official GDPa1 training CSV and ESM vectors inside the `data/` folder.

```bash
python -m adsentinel.train_cv \
  --train-csv data/GDPa1_v1.2_20250814.csv \
  --out-csv outputs/gdpa1_cv_predictions.csv
```

This will:

- Load GDPa1 training data
- Build three feature blocks (global, CDR zoom, ESM)
- Run Click nested CV to select best configuration per target
- Train Ridge → XGBoost ensemble with OOF stacking
- Print Spearman correlations per property
- Save out-of-fold predictions to CSV

---

## 4️⃣ Train on full dataset and generate heldout predictions

```bash
python -m adsentinel.predict \
  --train-csv data/GDPa1_v1.2_20250814.csv \
  --heldout-csv data/heldout-set-sequences.csv \
  --out-train-csv outputs/gdpa1_train_predictions.csv \
  --out-heldout-csv outputs/gdpa1_heldout_predictions.csv
```

This will:

- Run Click to select best config per target
- Train on the full training set with selected configs
- Generate predictions for the heldout/test set
- Export submission-ready CSV files

---

# Results

## Click CV (sequence + CDR zoom + ESM-2, adaptive selection)

Evaluated using the official GDPa1 cluster-aware fold column with nested CV for configuration selection.

| Property | Spearman ρ | Click config | p-value |
|-----------|-----------|--------------|---------|
| HIC | **0.428** | full_esm (5/5) | 3.3e-12 |
| AC-SINS (pH 7.4) | **0.370** | full_esm (4/5) | 3.0e-09 |
| PR_CHO | **0.429** | full_esm (5/5) | 3.1e-10 |
| Tm2 | **0.182** | full_esm (3/5), body20+pinze (2/5) | 1.1e-02 |
| Titer | **0.286** | full_esm (5/5) | 7.1e-06 |

All five properties show statistically significant ranking signal.

**Key finding**: With 246 samples, ESM embeddings dominate for most targets. The CDR zoom features show potential for Tm2 (where Click is split between configurations), suggesting that with larger datasets the zoom features would contribute more signal.

## Baseline comparison (sequence + isotype only, no ESM)

| Property | Spearman ρ |
|-----------|-----------|
| HIC | 0.304 |
| Tm2 | 0.251 |
| AC-SINS (pH 7.4) | 0.238 |
| PR_CHO | 0.177 |
| Titer | 0.091 (n.s.) |

## Heldout Test Set (competition submission)

| Property | Spearman ρ (heldout) | Comment |
|-----------|---------------------|----------|
| HIC | 0.495 | Strong sequence signal |
| Tm2 | 0.203 | Limited without 3D information |
| PR_CHO | 0.054 | Weak sequence signal |
| AC-SINS | 0.038 | Weak sequence signal |
| Titer | -0.028 | No detectable sequence signal |

---

## Model Architecture

AdSentinel uses a two-level architecture:

**Level 1 – Feature blocks**:
- Global (body): sequence composition, length, isotype
- Zoom (pinze): per-CDR hydropathy, charge patches, sticky regions via sliding window
- ESM: pre-trained protein language model embeddings (optionally PCA-compressed)

**Level 2 – Click selection**:
- Nested CV tests each feature combination
- Selects the best configuration per target property
- Adapts to dataset size (more features useful with more data)

**Level 3 – Regression**:
- Ridge regression produces baseline predictions
- XGBoost learns residual nonlinear structure
- OOF stacking prevents data leakage

---

## CDR Zoom Features

The zoom features extract local molecular descriptors from each CDR region (H1, H2, H3, L1, L2, L3) using AHo numbering alignment:

- **Length**: CDR loop length (especially CDR-H3, range 4–22 residues in GDPa1)
- **Hydropathy**: mean, max, std of Kyte-Doolittle scores per CDR
- **Charge**: net charge and absolute charge density per CDR (includes histidine)
- **Sticky patch**: maximum hydropathy in a sliding 5-residue window — detects the "stickiest" surface spot

These features model the antibody's "clamp" regions that drive non-specific interactions, self-association, and polyreactivity.

---

## Future Work

**More data**: The Click mechanism showed that CDR zoom features have potential for Tm2 but need more training samples to outperform raw ESM. With larger datasets (>1000 antibodies), the zoom features are expected to contribute significantly.

**3D structural extension**: A structural layer has been designed and validated on a single AlphaFold-Multimer structure. Planned features include radius of gyration, SASA, VH–VL interface compactness, pLDDT confidence metrics, and PAE cross-chain scores. See `docs/` for the full design document.

**Per-target model specialization**: The Click mechanism could be extended to also select different model hyperparameters (not just feature sets) per target.

---

# Repository Structure

```
AdSentinel/
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   └── adsentinel/
│       ├── __init__.py
│       ├── features.py        # three-block feature pipeline (global + zoom + ESM)
│       ├── model.py           # AdSentinelRegressor + ClickSelector
│       ├── train_cv.py        # Click cross-validation
│       └── predict.py         # Click train + heldout prediction
│
├── docs/
│   └── 3D_verification_and_theory.md # 3D structural extension design document
│
├── data/
│   ├── README.md              # instructions to obtain GDPa1
│   └── AbSentinel_vectors_1280.csv  # ESM-2 pre-computed embeddings
│
└── outputs/                   # generated prediction files (created at runtime)
```

---

# Reproducibility

The model uses:

- 5-fold cluster-aware cross-validation (official GDPa1 fold column)
- Nested CV for Click configuration selection (no data leakage)
- Spearman correlation as primary metric
- Deterministic random seeds
- Out-of-fold stacking to prevent leakage
- Explicit three-block feature computation pipeline

All results reported in this README are fully reproducible from the published code and data.

---

# Positioning

AdSentinel 2.7 is:

✅ Adaptive (Click selects features per target)

✅ Interpretable (CDR zoom features have biological meaning)

✅ Modular and reproducible

✅ Designed for extension (3D features, larger datasets)

❌ Not a 3D-aware physics model (yet)

It should be viewed as an **adaptive sequence baseline** for antibody developability ranking. The Click mechanism ensures the model uses the right level of detail for each property, and is designed to improve automatically as more training data becomes available.
