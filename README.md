AdSentinel 2.6 – Antibody Developability Model

AdSentinel 2.6 is a hybrid, sequence-based model for predicting antibody developability properties on the GDPa1 dataset (Ginkgo Antibody Developability Benchmark).

The model combines:

ESM-2 embeddings (mean-pooled VH/VL)

CDR-based biochemical descriptors (length, GRAVY, entropy, charge)

Global sequence biophysical features (hydrophobicity, net charge, aromaticity)

A two-stage regression ensemble: Ridge + XGBoost


This repository contains the exact implementation used for the sequence-only submission to the Ginkgo challenge.

> ⚠️ Note
A 3D structural extension (AlphaFold-based features: pLDDT, radius of gyration, PAE, etc.) was designed conceptually but not included in the competition submission.
This repo reflects the validated sequence-only baseline.




---

Target properties (GDPa1)

The model predicts five biophysical properties:

HIC – Hydrophobic Interaction Chromatography retention time

AC-SINS_pH7.4 – Self-association

PR_CHO – Polyreactivity in CHO cells

Tm2 – CH2 domain thermostability

Titer – Expression yield



---

Results – Ginkgo 2025 Challenge (Heldout Test Set)

Property	Spearman ρ (heldout)	Comment

Hydrophobicity (HIC)	0.495	Strong sequence signal
Thermostability (Tm2)	0.203	Limited without 3D info
Polyreactivity (PR_CHO)	0.054	Weak signal
Self-association (AC-SINS)	0.038	Weak signal
Titer	-0.028	No detectable signal


Interpretation

HIC is largely sequence-driven → AdSentinel 2.6 performs competitively.

Other properties depend strongly on 3D packing, interface physics, and experimental variance → sequence-only models are inherently limited.

This repository is intentionally transparent: it provides a solid baseline, not an inflated claim.



---

Repository structure (current)

AdSentinel/
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   └── adsentinel/
│       ├── __init__.py
│       ├── features.py        # sequence + CDR feature engineering
│       ├── esm_embeddings.py  # ESM-2 embedding wrapper
│       ├── model.py           # Ridge + XGBoost ensemble
│       ├── train_cv.py        # 5-fold CV on GDPa1
│       └── predict.py         # heldout prediction pipeline
│
├── configs/
│   └── gdpa1_config.yaml      # dataset paths and hyperparameters
│
├── data/
│   └── README.md              # instructions to obtain GDPa1
│
├── notebooks/
│   └── 01_exploration.ipynb   # optional analysis
│
├── reports/
│   └── AdSentinel2.6_report.pdf
│
└── scripts/
    ├── run_cv.sh
    └── run_heldout.sh


---

Reproducibility

The model uses:

5-fold stratified cross-validation (GDPa1 fold column)

Spearman correlation as primary metric

Deterministic seeds for reproducibility


Heldout predictions follow the official GDPa1 submission format.


---

Positioning

AdSentinel 2.6 is:

✅ Interpretable

✅ Fast to train

✅ Fully sequence-based

❌ Not a 3D-aware physics model


It is best viewed as a robust sequence baseline for developability modeling.

## Repository structure

This section reflects the current structure of the AdSentinel repository
as used in the Ginkgo GDPa1 competition submission.

```text
AdSentinel/
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   └── adsentinel/
│       ├── __init__.py
│       ├── features.py          # Sequence + CDR feature engineering
│       ├── esm_embeddings.py    # ESM-2 embedding wrapper
│       ├── model.py             # Ridge + XGBoost ensemble
│       ├── train_cv.py          # 5-fold cross-validation on GDPa1
│       └── predict.py           # Heldout prediction pipeline
│
├── configs/
│   └── gdpa1_config.yaml        # Dataset paths and hyperparameters
│
├── data/
│   ├── README.md                # Instructions to obtain GDPa1
│   └── .gitkeep
│
├── notebooks/
│   └── 01_exploration.ipynb     # Optional exploratory analysis
│
├── reports/
│   └── AdSentinel_November_2025.pdf
│
└── scripts/
    ├── run_cv.sh                # Helper script for CV
    └── run_heldout.sh           # Helper script for heldout predictions
