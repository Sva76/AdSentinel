# AdSentinel 2.6 – Antibody Developability Model

AdSentinel 2.6 is a sequence-based model for predicting antibody developability properties on the **GDPa1 dataset** (Ginkgo Antibody Developability Benchmark).

The model combines:

- **Global sequence biophysical features** (hydrophobicity fraction, net charge, sequence length)
- **Isotype-aware features** (`hc_subtype` one-hot encoding)
- A **two-stage regression ensemble**: Ridge → XGBoost (with out-of-fold stacking)

This repository contains the exact implementation used for the **sequence-only submission** to the Ginkgo challenge.

> ⚠️ **Note**  
> ESM-2 embeddings were used during development and contributed to higher scores,
> but the embedding vectors are not currently included in this repository due to file size constraints.
> The published code runs with physicochemical + isotype features only.
>
> A 3D structural extension (AlphaFold-based features: pLDDT, radius of gyration, PAE, VH–VL interface metrics)
> has been designed and validated on a single antibody structure but was not included in the competition submission.
> See the `docs/` folder for the full 3D design document.

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

## 3️⃣ Run 5-fold cross-validation (GDPa1)

Place the official GDPa1 training CSV inside the `data/` folder.

```bash
python -m adsentinel.train_cv \
  --train-csv data/GDPa1_v1.2_20250814.csv \
  --out-csv outputs/gdpa1_cv_predictions.csv
```

This will:

- Load GDPa1 training data
- Compute sequence + isotype features
- Train the Ridge + XGBoost ensemble
- Perform 5-fold cluster-aware cross-validation
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

- Train on the full training set
- Generate predictions for the heldout/test set
- Export submission-ready CSV files

---

# Results

## Cluster-Aware Cross-Validation (published code, sequence + isotype features)

Evaluated using the official GDPa1 cluster-aware fold column
(`hierarchical_cluster_IgG_isotype_stratified_fold`), which prevents similar
antibodies from appearing in both training and test sets.

| Property | Spearman ρ | p-value |
|-----------|-----------|---------|
| HIC | **0.304** | 1.4e-06 |
| Tm2 | **0.251** | 4.2e-04 |
| AC-SINS (pH 7.4) | **0.238** | 1.9e-04 |
| PR_CHO | **0.177** | 1.3e-02 |
| Titer | 0.091 | 0.16 (n.s.) |

Four out of five properties show statistically significant ranking signal
using only 9 features (6 physicochemical + 3 isotype one-hot).

## Heldout Test Set (competition submission, included ESM-2 embeddings)

The competition submission used ESM-2 (mean-pooled VH/VL) embeddings in addition
to the features above. These results are **not reproducible** from the current
published code alone.

| Property | Spearman ρ (heldout) | Comment |
|-----------|---------------------|----------|
| HIC | 0.495 | Strong sequence signal |
| Tm2 | 0.203 | Limited without 3D information |
| PR_CHO | 0.054 | Weak sequence signal |
| AC-SINS | 0.038 | Weak sequence signal |
| Titer | -0.028 | No detectable sequence signal |

---

## Model Architecture

AdSentinel uses a lightweight stacked regression:

1. **Ridge regression** produces baseline predictions from sequence features.
2. **XGBoost** learns residual structure using the original features plus Ridge predictions.

To prevent data leakage during stacking, Ridge predictions used to train XGBoost
are generated using **out-of-fold cross-validation** (not in-sample predictions).

---

## Future Work

A structural extension of AdSentinel has been designed and partially validated.

Planned structural features include:

- AlphaFold-Multimer predicted structures
- Radius of gyration
- Solvent accessible surface area (SASA)
- VH–VL interface compactness (contact count, mean Cα distance)
- pLDDT confidence metrics (mean, min, std)
- PAE cross-chain scores (VH→VL interface confidence)

Validation on a single antibody structure confirmed that all features are
extractable and produce biophysically meaningful values. Full-dataset
validation requires generating AlphaFold-Multimer structures for all 246
antibodies in the GDPa1 training set.

These features are expected to improve prediction for properties influenced by
structural stability and surface interactions, particularly **Tm2** and **AC-SINS**.

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
│       ├── features.py        # sequence + isotype feature extraction
│       ├── model.py           # Ridge + XGBoost ensemble (OOF stacking)
│       ├── train_cv.py        # cluster-aware cross-validation
│       └── predict.py         # full training + heldout prediction
│
├── docs/
│   └── adsentinel_3d_guide.md # 3D structural extension design document
│
├── data/
│   └── README.md              # instructions to obtain GDPa1
│
└── outputs/                   # generated prediction files (created at runtime)
```

---

# Reproducibility

The model uses:

- 5-fold cluster-aware cross-validation (official GDPa1 fold column)
- Spearman correlation as primary metric
- Deterministic random seeds
- Out-of-fold stacking to prevent leakage
- Explicit feature computation pipeline

Heldout predictions follow the official GDPa1 submission format.

---

# Positioning

AdSentinel 2.6 is:

✅ Interpretable  
✅ Fully sequence-based  
✅ Modular and reproducible  
✅ Easy to extend (ESM embeddings, structural features)  
❌ Not a 3D-aware physics model (yet)  

It should be viewed as a **sequence baseline** for antibody developability
ranking, suitable as a fast computational pre-filter in antibody selection
pipelines.
```
