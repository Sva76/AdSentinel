# AdSentinel 2.6 – Antibody Developability Model

AdSentinel 2.6 is a hybrid, sequence-based model for predicting antibody developability properties on the **GDPa1 dataset** (Ginkgo Antibody Developability Benchmark).

The model combines:

- **ESM-2 embeddings** (mean-pooled VH/VL sequences)
- **CDR-based biochemical descriptors** (length-based features from AHo numbering)
- **Global sequence biophysical features** (hydrophobicity fraction, net charge, sequence length)
- A **two-stage regression ensemble**: Ridge → XGBoost

This repository contains the exact implementation used for the **sequence-only submission** to the Ginkgo challenge.

> ⚠️ **Note**  
> A 3D structural extension (AlphaFold-based features: pLDDT, radius of gyration, PAE, etc.) was designed conceptually but was not included in the competition submission.  
> This repository reflects the validated **sequence-only baseline**.

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
- Compute sequence + CDR + ESM features
- Train the Ridge + XGBoost ensemble
- Perform 5-fold stratified cross-validation
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

# Results – Ginkgo 2025 Challenge (Heldout Test Set)

| Property | Spearman ρ (heldout) | Comment |
|-----------|---------------------|----------|
| HIC | 0.495 | Strong sequence signal |
| Tm2 | 0.203 | Limited without 3D information |
| PR_CHO | 0.054 | Weak sequence signal |
| AC-SINS | 0.038 | Weak sequence signal |
| Titer | -0.028 | No detectable sequence signal |

### Interpretation

- **HIC is largely sequence-driven** → AdSentinel 2.6 performs competitively.
- Other properties depend strongly on 3D packing, interface physics, and experimental variance → sequence-only models are inherently limited.

This repository is intentionally transparent:  
it provides a solid and reproducible **sequence baseline**, not an inflated claim.

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
│       ├── features.py        # sequence + CDR + ESM feature extraction
│       ├── esm_utils.py       # ESM-2 embedding wrapper
│       ├── model.py           # Ridge + XGBoost ensemble
│       ├── train_cv.py        # 5-fold cross-validation
│       └── predict.py         # full training + heldout prediction
│
├── data/
│   └── README.md              # instructions to obtain GDPa1
│
└── outputs/                   # generated prediction files (created at runtime)
```

---

# Reproducibility

The model uses:

- 5-fold stratified cross-validation (official GDPa1 fold column)
- Spearman correlation as primary metric
- Deterministic random seeds
- Explicit feature computation pipeline

Heldout predictions follow the official GDPa1 submission format.

---

# Positioning

AdSentinel 2.6 is:

✅ Interpretable  
✅ Fully sequence-based  
✅ Modular and reproducible  
✅ Easy to extend  
❌ Not a 3D-aware physics model  

It should be viewed as a **robust sequence baseline** for antibody developability modeling.

The feature pipeline now supports **isotype-aware modeling**
via `hc_subtype` one-hot encoding, following feedback from
the Ginkgo AbDev competition team that antibody subclass
information can influence model performance.
