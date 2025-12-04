-----------------------------------------

AdSentinel – 3D Structural Verification Guide (Technical Note)

Extending the AdSentinel Framework with Structural Bioinformatics

--------------------------------------------------------

This document describes how a fully functional 3D-enabled version of AdSentinel should validate, extract, and integrate structural features derived from AlphaFold2/ESMFold predictions, and how these features interact mathematically with the 1D (sequence) feature pipeline.

The goal is to provide a complete roadmap for implementing AdSentinel-3D, independent of missing compute or incomplete structural datasets.


---

1. Overview

AdSentinel currently implements:

1. Sequence-level engineered features


2. CDR segmentation (AHo numbering)


3. ESM-2 embeddings (mean pooled)


4. Ridge/XGBoost meta-regressor



The original design also required a 3D validation layer, not yet fully executed during the competition due to missing structural files and compute constraints.
This note formalizes:

how structures must be validated

what geometric descriptors must be extracted

how they integrate mathematically with ESM / CDR features

how they should be used during training and inference



---

2. Required Inputs for 3D Processing

To complete the AdSentinel 3D pipeline, each antibody must have:

/structures/{antibody_name}/
    model.pdb                   # AlphaFold/ESMFold model
    plddt_scores.json           # confidence metrics
    pae.json                    # Predicted Aligned Error matrix

If a structure is missing, a masking strategy is applied:

3D_features = NaN
feature_mask_3D = 0

Otherwise:

feature_mask_3D = 1

This ensures that the regression model never assumes the presence of structural data.


---

3. Mathematical Core of AdSentinel-3D

Below are the mathematical definitions used to compute 3D descriptors.


---

3.1 Radius of Gyration

Given atom coordinates :

R_g = \sqrt{ \frac{1}{N} \sum_{i=1}^N \lVert x_i - \bar{x} \rVert^2 }

where  is the centroid.


---

3.2 pLDDT Metrics

\text{pLDDT}_{mean} = \frac{1}{L} \sum_{i=1}^{L} \text{pLDDT}_i

\text{pLDDT}_{min} = \min_i \{ \text{pLDDT}_i \}

\text{pLDDT}_{std} = \sqrt{ \frac{1}{L} \sum ( \text{pLDDT}_i - \text{mean} )^2 }


---

3.3 Interface Scores (VH–VL interface compactness)

Using Cα atoms from VH and VL:

d_{interface} = \frac{1}{|H||L|} \sum_{h \in VH} \sum_{l \in VL} \lVert x_h - x_l \rVert

Lower values → stronger packing.


---

3.4 Solvent Accessibility (SASA)

For residue :

SASA_i = \text{Shrake-Rupley}(x_i)

Global metrics:

SASA_{mean},\; SASA_{CDR-H3},\; SASA_{paratope}


---

3.5 Structural Instability Score

A composite index:

S_{instability} = w_1 R_g + w_2 d_{interface} + w_3 (1 - \text{pLDDT}_{mean}/100)

Weights tuned via cross-validation.


---

4. Integration with Sequence Features

After sequence feature extraction:

F_seq   = [hydrophobicity, charge, CDR lengths, ...]
F_esm   = [ESM pooled embedding]
F_3D    = [Rg, interface distance, SASA, pLDDT, ...]
M_3D    = binary mask (1 if structure exists)

The final feature vector is:

X = [F_{seq} \; || \; F_{esm} \; || \; F_{3D} \; || \; M_{3D}]

Masking ensures that the model learns to ignore missing structures properly.


---

5. AdSentinel-3D Algorithm (Pseudocode)

def AdSentinel3D(df):
    # 1. Sequence features
    seq = compute_sequence_features(df)

    # 2. CDR extraction
    cdr = extract_cdrs(df)

    # 3. ESM embeddings
    emb = [esm_embed_sequence(vh + vl) for vh,vl in df[['vh','vl']].values]

    # 4. Structural features
    struct = []
    mask = []
    for name in df.antibody_name:
        if structure_exists(name):
            struct.append(extract_3d_features(name))
            mask.append(1)
        else:
            struct.append(nan_vector)
            mask.append(0)

    # 5. Feature concatenation
    X = concatenate([seq, cdr, emb, struct, mask])

    return X


---

6. How to Validate 3D Structures (Checklist)

A structure is valid if:

✅ 1. File is present

model.pdb exists and loads without parse errors.

✅ 2. Chain identification is correct

VH and VL must be distinguishable.

✅ 3. pLDDT quality threshold

Mean pLDDT > 70 is recommended.

✅ 4. No extreme distortions

Radius of gyration must satisfy:

20 \text{ Å} < R_g < 60 \text{ Å}

✅ 5. No missing residues in CDR loops

CDR-H3 must be fully resolved.

If any condition fails → M_3D = 0, F_3D = NaN.


---

7. Recommended Use in Training

Normalize 3D features separately

Use PCA for embeddings (optional)

Use Ridge or ElasticNet (3D features are often correlated)

Use masking to prevent model collapse when 3D is missing



---

8. Conclusion

This 3D validation and extraction framework turns AdSentinel into a hybrid sequence–structure predictor, enabling:

better Tm2 prediction

improved self-association modeling

robust generalization across antibody families


This document completes the conceptual design originally intended for AdSentinel and makes the repository fully self-contained for future development or collaboration.


---

9. Minimal References Actually Used

Only the sources truly required for the implemented work:

1. AHo numbering
Honegger & Plückthun, J Mol Biol (2001)
– Used for CDR segmentation.


2. ESM-2 embeddings
Lin et al., Nature (2023)
– Used for sequence embedding.


3. AlphaFold pLDDT / PAE
Jumper et al., Nature (2021)
– Basis for structural scores used in this document.



No fictional or unused references included.
