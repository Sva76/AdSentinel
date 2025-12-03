# AdSentinel 2.6 – Antibody Developability Model

AdSentinel 2.6 is a hybrid, sequence-based model for predicting antibody developability
properties on the **GDPa1** dataset (Ginkgo Antibody Developability Benchmark).

The model combines:

- **ESM-2 embeddings** (mean-pooled VH/VL)
- **CDR-based biochemical descriptors** (length, GRAVY, entropy, charge)
- **Global sequence biophysical features** (hydrophobicity, net charge, aromaticity)
- A **two-stage regression**: Ridge + XGBoost ensemble

> ⚠️ Note  
> The original design also included a 3D structural module (AlphaFold-based features:
> radius of gyration, pLDDT, PAE, etc.), but this part was **not implemented** in the
> competition submission. The code in this repository reflects the **sequence-only
> version** used in the Ginkgo challenge.

---

## Target properties (GDPa1)

The model predicts 5 biophysical properties:

- **HIC** – Hydrophobic Interaction Chromatography retention time  
- **AC-SINS_pH7.4** – Self-association  
- **PR_CHO** – Polyreactivity in CHO cells  
- **Tm2** – CH2 domain thermostability  
- **Titer** – Expression yield  

---

## Results (Ginkgo 2025 Challenge – Heldout test set)

This repository corresponds to the **AdSentinel 2.6** submission to the  
Ginkgo Antibody Developability Challenge (GDPa1).  
The heldout (private test set) Spearman correlations were:

| Property         | Spearman ρ (heldout) | Comment                            |
|------------------|----------------------|------------------------------------|
| Hydrophobicity   | **0.495**           | Strong signal from sequence       |
| Thermostability  | 0.203               | Limited (no 3D structural info)   |
| Polyreactivity   | 0.054               | Almost flat                        |
| Self-association | 0.038               | Almost flat                        |
| Titer            | -0.028              | No signal                          |

Interpretation:

- **HIC** is largely sequence-driven → AdSentinel 2.6 performs reasonably well.
- **Tm2, PR_CHO, AC-SINS, Titer** are strongly influenced by 3D packing,  
  isotype-dependent physics, and experimental noise → without a 3D module,  
  performance is limited.

This repository is therefore **honest and transparent**:  
it contains a **good sequence-only baseline**, but not the full 3D-aware design.

---

## Repository structure

Planned structure:

```text
AdSentinel-2.6/
├── README.md                # This file
├── LICENSE                  # e.g. Apache-2.0
├── .gitignore               # Python + notebooks + data
├── requirements.txt         # Python deps (sklearn, xgboost, transformers, etc.)
│
├── src/
│   └── adsentinel/
│       ├── __init__.py
│       ├── features.py      # sequence + CDR features
│       ├── esm_embeddings.py# ESM-2 embedding wrapper
│       ├── model.py         # Ridge + XGBoost ensemble
│       ├── train_cv.py      # 5-fold CV on GDPa1
│       └── predict.py       # heldout prediction script
│
├── configs/
│   └── gdpa1_config.yaml    # paths, hyperparameters
│
├── data/
│   ├── README.md            # how to download GDPa1
│   └── .gitkeep             # keep folder in git
│
├── notebooks/
│   └── 01_exploration.ipynb # optional, data exploration
│
├── reports/
│   └── AdSentinel2.6_report.pdf   # Zenodo-style technical report
│
└── scripts/
    ├── run_cv.sh            # bash helper to run CV
    └── run_heldout.sh       # bash helper to run heldout prediction
